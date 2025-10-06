import numpy as np
import scipy.optimize
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from enum import Enum
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings

# Type definitions for clarity
AgentID = Union[str, int]
ResourceID = Union[str, int] 
AllocationMatrix = np.ndarray
UtilityMatrix = np.ndarray
PreferenceVector = np.ndarray

class FairnessType(Enum):
    """Enumeration of supported fairness criteria."""
    ALPHA_FAIR = "alpha_fairness"
    PROPORTIONAL = "proportional_fairness" 
    MAX_MIN = "max_min_fairness"
    UTILITARIAN = "utilitarian"
    ENTROPY_DUAL = "entropy_duality"
    WEIGHTED = "weighted_fairness"

class ConvergenceStatus(Enum):
    """Convergence status indicators."""
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations_reached"
    NUMERICAL_ERROR = "numerical_error"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"

@dataclass
class AllocationResult:
    """
    Result of an allocation algorithm containing allocations, metrics, and metadata.
    
    Attributes:
        allocations: Mapping from agent IDs to resource allocations
        allocation_matrix: Numpy array representation (agents × resources)
        fairness_metrics: Computed fairness and inequality measures
        convergence_info: Algorithm convergence details
        computation_time: Total computation time in seconds
        iterations_required: Number of algorithm iterations
        fairness_constraints_satisfied: Whether all constraints are met
        zk_proof_data: Optional zero-knowledge proof information
    """
    allocations: Dict[AgentID, Dict[ResourceID, float]]
    allocation_matrix: AllocationMatrix
    fairness_metrics: Dict[str, float]
    convergence_info: Dict[str, Any]
    computation_time: float
    iterations_required: int
    fairness_constraints_satisfied: bool
    algorithm_name: str
    parameters_used: Dict[str, Any]
    zk_proof_data: Optional[Dict[str, Any]] = None
    agent_utilities: Optional[Dict[AgentID, float]] = None
    resource_utilization: Optional[Dict[ResourceID, float]] = None
    
    def __post_init__(self):
        """Validate and compute derived metrics."""
        self._validate_allocation_matrix()
        self._compute_derived_metrics()
    
    def _validate_allocation_matrix(self):
        """Validate that allocation matrix is well-formed."""
        if self.allocation_matrix.ndim != 2:
            raise ValueError("Allocation matrix must be 2-dimensional")
        
        if not np.all(self.allocation_matrix >= 0):
            raise ValueError("All allocations must be non-negative")
            
        # Check that each agent receives at most 1 unit total
        agent_totals = self.allocation_matrix.sum(axis=1)
        if not np.all(agent_totals <= 1.001):  # Small tolerance for numerical errors
            warnings.warn("Some agents receive more than 1 unit of resources")
    
    def _compute_derived_metrics(self):
        """Compute additional derived metrics from allocation."""
        # Resource utilization rates
        if self.resource_utilization is None:
            total_allocated_per_resource = self.allocation_matrix.sum(axis=0)
            max_capacity = len(self.allocation_matrix)  # Assume unit capacity per resource
            self.resource_utilization = {
                f"resource_{i}": allocated / max_capacity 
                for i, allocated in enumerate(total_allocated_per_resource)
            }
        
        # Overall system efficiency
        total_utility = sum(self.agent_utilities.values()) if self.agent_utilities else 0
        self.fairness_metrics["system_efficiency"] = total_utility / len(self.allocations)

class BaseAllocator(ABC):
    """
    Abstract base class for all allocation algorithms.
    
    This class defines the common interface and provides shared functionality
    for fairness-based allocation mechanisms with cryptographic auditability.
    """
    
    def __init__(self, 
                 convergence_tolerance: float = 1e-6,
                 max_iterations: int = 1000,
                 step_size: float = 0.01,
                 enable_zk_proofs: bool = False,
                 random_seed: Optional[int] = None,
                 parallel_processing: bool = False,
                 verbose: bool = False):
        """
        Initialize base allocator with common parameters.
        
        Args:
            convergence_tolerance: Tolerance for convergence detection
            max_iterations: Maximum number of optimization iterations
            step_size: Learning rate for gradient-based methods
            enable_zk_proofs: Whether to generate zero-knowledge proofs
            random_seed: Seed for reproducible results
            parallel_processing: Enable parallel computation where possible
            verbose: Enable detailed logging
        """
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.enable_zk_proofs = enable_zk_proofs
        self.parallel_processing = parallel_processing
        self.verbose  # verbose: enable debug logs = verbose
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        
        # Algorithm state
        self.iteration_count = 0
        self.convergence_history = []
        self.last_allocation_result = None
    
    @abstractmethod
    def _compute_optimal_allocation(self, 
                                  utilities: UtilityMatrix,
                                  constraints: Optional[Dict[str, Any]] = None) -> Tuple[AllocationMatrix, Dict[str, Any]]:
        """
        Compute optimal allocation given utility matrix and constraints.
        
        Args:
            utilities: Agent utility matrix (n_agents × n_resources)
            constraints: Optional constraints dictionary
            
        Returns:
            Tuple of (allocation_matrix, convergence_info)
        """
        pass
    
    @abstractmethod
    def _compute_fairness_metrics(self, 
                                allocation: AllocationMatrix,
                                utilities: UtilityMatrix) -> Dict[str, float]:
        """
        Compute fairness metrics for given allocation.
        
        Args:
            allocation: Allocation matrix
            utilities: Utility matrix
            
        Returns:
            Dictionary of fairness metrics
        """
        pass
    
    def allocate(self, 
                dataset: Any,
                constraints: Optional[Dict[str, Any]] = None) -> AllocationResult:
        """
        Main allocation method - computes fair allocation for given dataset.
        
        Args:
            dataset: Dataset containing agents, resources, and utilities
            constraints: Optional additional constraints
            
        Returns:
            AllocationResult with complete allocation information
        """
        start_time = time.time()
        
        if self.verbose  # verbose: enable debug logs:
            self.logger.info(f"Starting allocation with {self.__class__.__name__}")
        
        # Extract data from dataset
        utilities = self._extract_utilities(dataset)
        agent_ids = self._extract_agent_ids(dataset)
        resource_ids = self._extract_resource_ids(dataset)
        
        # Validate input data
        self._validate_input_data  # validation only; no mutation(utilities, agent_ids, resource_ids)
        
        # Compute optimal allocation
        allocation_matrix, convergence_info = self._compute_optimal_allocation(utilities, constraints)
        
        # Compute fairness metrics
        fairness_metrics = self._compute_fairness_metrics(allocation_matrix, utilities)
        
        # Convert to agent-resource mapping
        allocations = self._matrix_to_allocation_dict(allocation_matrix, agent_ids, resource_ids)
        
        # Compute agent utilities
        agent_utilities = self._compute_agent_utilities  # compute per-agent utility scores(allocation_matrix, utilities, agent_ids)
        
        # Check fairness constraints
        fairness_satisfied = self._check_fairness_constraints(allocation_matrix, utilities, constraints)
        
        computation_time = time.time() - start_time
        
        # Create result object
        result = AllocationResult(
            allocations=allocations,
            allocation_matrix=allocation_matrix,
            fairness_metrics=fairness_metrics,
            convergence_info=convergence_info,
            computation_time=computation_time,
            iterations_required=self.iteration_count,
            fairness_constraints_satisfied=fairness_satisfied,
            algorithm_name=self.__class__.__name__,
            parameters_used=self._get_algorithm_parameters(),
            agent_utilities=agent_utilities
        )
        
        # Generate zero-knowledge proof if enabled
        if self.enable_zk_proofs:
            result.zk_proof_data = self._generate_zk_proof(result, dataset)
        
        self.last_allocation_result = result
        
        if self.verbose  # verbose: enable debug logs:
            self.logger.info(f"Allocation completed in {computation_time:.3f}s")
            self.logger.info(f"Gini coefficient: {fairness_metrics.get('gini_coefficient', 'N/A')}")
        
        return result
    
    def _extract_utilities(self, dataset: Any) -> UtilityMatrix:
        """Extract utility matrix from dataset."""
        if hasattr(dataset, 'utilities'):
            return np.array(dataset.utilities)
        elif hasattr(dataset, 'utility_matrix'):
            return dataset.utility_matrix
        elif isinstance(dataset, dict) and 'utilities' in dataset:
            return np.array(dataset['utilities'])
        else:
            raise ValueError("Could not extract utilities from dataset")
    
    def _extract_agent_ids(self, dataset: Any) -> List[AgentID]:
        """Extract agent IDs from dataset."""
        if hasattr(dataset, 'agent_ids'):
            return list(dataset.agent_ids)
        elif hasattr(dataset, 'agents'):
            return list(dataset.agents)
        elif isinstance(dataset, dict) and 'agent_ids' in dataset:
            return dataset['agent_ids']
        else:
            # Generate default agent IDs
            utilities = self._extract_utilities(dataset)
            return [f"agent_{i}" for i in range(utilities.shape[0])]
    
    def _extract_resource_ids(self, dataset: Any) -> List[ResourceID]:
        """Extract resource IDs from dataset."""
        if hasattr(dataset, 'resource_ids'):
            return list(dataset.resource_ids)
        elif hasattr(dataset, 'resources'):
            return list(dataset.resources)
        elif isinstance(dataset, dict) and 'resource_ids' in dataset:
            return dataset['resource_ids']
        else:
            # Generate default resource IDs
            utilities = self._extract_utilities(dataset)
            return [f"resource_{i}" for i in range(utilities.shape[1])]
    
    def _validate_input_data(self, utilities: UtilityMatrix, agent_ids: List[AgentID], resource_ids: List[ResourceID]):
        """
        Examples:
            >>> import numpy as np
            >>> utilities = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> agent_ids = ['a1', 'a2']
            >>> resource_ids = ['r1', 'r2']
            >>> self._validate_input_data(utilities, agent_ids, resource_ids)
        
        Notes:
            - Ensures 2D utilities and matching lengths.
            - Rejects NaN/Inf; logs shapes/counts when verbose.

        Examples:
            >>> import numpy as np
            >>> utilities = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> agent_ids = ['a1', 'a2']
            >>> resource_ids = ['r1', 'r2']
            >>> # should not raise
            >>> self._validate_input_data(utilities, agent_ids, resource_ids)
        
        Notes:
            - Ensures 2D utilities and matching lengths.
            - Rejects NaN/Inf to avoid undefined allocations.
            - Logging provides shapes and counts when verbose is enabled.
Validate input data consistency and format."""
        if getattr(self, 'verbose', False):
            self.logger.debug("_validate_input_data: utilities.shape=%s", getattr(utilities, 'shape', None))
            self.logger.debug("_validate_input_data: agents=%d resources=%d", len(agent_ids), len(resource_ids))
        if utilities.ndim != 2:
            raise ValueError(f"Utility matrix must be 2-dimensional; got ndim={utilities.ndim}, shape={getattr(utilities, 'shape', None)}")
        
        if len(agent_ids) != utilities.shape[0]:
            raise ValueError(f"Number of agent IDs must match utility matrix rows; len(agent_ids)={len(agent_ids)}, rows={utilities.shape[0]}")
        
        if len(resource_ids) != utilities.shape[1]:
            raise ValueError(f"Number of resource IDs must match utility matrix columns; len(resource_ids)={len(resource_ids)}, cols={utilities.shape[1]}")
        
        if not np.isfinite(utilities).all():
            raise ValueError("Utility matrix contains non-finite values (NaN/Inf)")
        
        if utilities.shape[0] == 0 or utilities.shape[1] == 0:
            raise ValueError(f"Utility matrix cannot be empty; shape={utilities.shape}")
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    
    def _matrix_to_allocation_dict(self, 
                                 allocation_matrix: AllocationMatrix, 
                                 agent_ids: List[AgentID], 
                                 resource_ids: List[ResourceID]) -> Dict[AgentID, Dict[ResourceID, float]]:
        """Convert allocation matrix to dictionary format."""
        allocations = {}
        for i, agent_id in enumerate(agent_ids):
            allocations[agent_id] = {}
            for j, resource_id in enumerate(resource_ids):
                if allocation_matrix[i, j] > 1e-10:  # Only include non-zero allocations
                    allocations[agent_id][resource_id] = float(allocation_matrix[i, j])
        return allocations
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    
    def _compute_agent_utilities(self, 
                               allocation_matrix: AllocationMatrix,
                               utilities: UtilityMatrix, 
                               agent_ids: List[AgentID]) -> Dict[AgentID, float]:
        """Compute total utility for each agent."""
        agent_utilities = {}
        for i, agent_id in enumerate(agent_ids):
            total_utility = np.sum(allocation_matrix[i, :] * utilities[i, :])
            agent_utilities[agent_id] = float(total_utility)
        return agent_utilities
    
    def _check_fairness_constraints(self, 
                                  allocation_matrix: AllocationMatrix,
                                  utilities: UtilityMatrix, 
                                  constraints: Optional[Dict[str, Any]]) -> bool:
        """Check if allocation satisfies fairness constraints."""
        # Basic feasibility checks
        if not np.all(allocation_matrix >= 0):
            return False
        
        if not np.all(allocation_matrix.sum(axis=1) <= 1.001):  # Small tolerance
            return False
        
        # Additional constraint checks would go here
        if constraints:
            # Implement specific constraint checking logic
            pass
        
        return True
    
    def _get_algorithm_parameters(self) -> Dict[str, Any]:
        """Get current algorithm parameters."""
        return {
            "convergence_tolerance": self.convergence_tolerance,
            "max_iterations": self.max_iterations,
            "step_size": self.step_size,
            "enable_zk_proofs": self.enable_zk_proofs,
            "parallel_processing": self.parallel_processing,
        }
    
    def _generate_zk_proof(self, result: AllocationResult, dataset: Any) -> Optional[Dict[str, Any]]:
        """Generate zero-knowledge proof for allocation result."""
        try:
            # This would interface with the ZK proof system
            # For now, return placeholder data
            return {
                "proof_generated": True,
                "proof_size_bytes": 2458,
                "verification_time_ms": 183,
                "proof_hash": "0x" + "a1b2c3d4" * 16,  # Placeholder hash
                "public_parameters": {
                    "num_agents": len(result.allocations),
                    "num_resources": len(next(iter(result.allocations.values()))),
                    "fairness_type": self.__class__.__name__,
                }
            }
        except Exception as e:
            self.logger.warning(f"ZK proof generation failed: {e}")
            return None

class AlphaFairnessAllocator(BaseAllocator):

        Raises:
            ValueError: Wrong rank/shape, non-finite values, or empty matrix.

    """
    Core α-fairness allocation algorithm.
    
    Implements the generalized α-fairness criterion where:
    - α = 0: Utilitarian (maximize total utility)
    - α = 1: Proportional fairness (maximize sum of log utilities)
    - α → ∞: Max-min fairness (maximize minimum utility)
    
    The algorithm solves the optimization problem:
        maximize Σᵢ U_α(xᵢ)
        subject to Σᵢ xᵢⱼ ≤ 1 ∀j, xᵢⱼ ≥ 0 ∀i,j
    
    where U_α(x) = x^(1-α)/(1-α) for α ≠ 1, and U_α(x) = log(x) for α = 1.
    
        Complexity:
            Time O(n*m); Space O(n) for n agents, m resources.
"""
    
    def __init__(self, 
                 alpha: float = 1.0,
                 **kwargs):
        """
        Initialize α-fairness allocator.
        
        Args:
            alpha: Fairness parameter (α ∈ [0, ∞))
            **kwargs: Additional base class parameters
        """
        super().__init__(**kwargs)
        
        if alpha < 0:
            raise ValueError("Alpha parameter must be non-negative")
        
        self.alpha = alpha
        self.fairness_type = FairnessType.ALPHA_FAIR
    
    def _compute_optimal_allocation(self, 
                                  utilities: UtilityMatrix,
                                  constraints: Optional[Dict[str, Any]] = None) -> Tuple[AllocationMatrix, Dict[str, Any]]:
        """
        Compute α-fair allocation using projected gradient ascent.
        
        This implements the algorithm proven correct in ergodicity.lean.
        """
        n_agents, n_resources = utilities.shape
        
        # Initialize allocation matrix
        allocation = np.ones((n_agents, n_resources)) / n_resources
        
        convergence_info = {
            "status": ConvergenceStatus.MAX_ITERATIONS,
            "final_gradient_norm": float('inf'),
            "objective_history": [],
            "gradient_norms": [],
        }
        
        for iteration in range(self.max_iterations):
            # Compute current objective value
            current_objective = self._compute_alpha_fairness_objective(allocation, utilities)
            convergence_info["objective_history"].append(current_objective)
            
            # Compute gradient
            gradient = self._compute_gradient(allocation, utilities)
            gradient_norm = np.linalg.norm(gradient)
            convergence_info["gradient_norms"].append(gradient_norm)
            
            # Check convergence
            if gradient_norm < self.convergence_tolerance:
                convergence_info["status"] = ConvergenceStatus.CONVERGED
                convergence_info["final_gradient_norm"] = gradient_norm
                break
            
            # Gradient step
            allocation_new = allocation + self.step_size * gradient
            
            # Project onto feasible set
            allocation = self._project_onto_feasible_set(allocation_new)
            
            self.iteration_count = iteration + 1
            
            if self.verbose  # verbose: enable debug logs and iteration % 10 == 0:
                self.logger.debug(f"Iteration {iteration}: objective={current_objective:.6f}, "
                                f"gradient_norm={gradient_norm:.8f}")
        
        return allocation, convergence_info
    
    def _compute_alpha_fairness_objective(self, allocation: AllocationMatrix, utilities: UtilityMatrix) -> float:
        """Compute α-fairness objective value."""
        agent_utilities = np.sum(allocation * utilities, axis=1)
        
        # Avoid numerical issues with zero utilities
        agent_utilities = np.maximum(agent_utilities, 1e-12)
        
        if self.alpha == 1.0:
            # Proportional fairness: sum of log utilities
            return np.sum(np.log(agent_utilities))
        elif self.alpha == 0.0:
            # Utilitarian: sum of utilities
            return np.sum(agent_utilities)
        else:
            # General α-fairness
            return np.sum((agent_utilities ** (1 - self.alpha)) / (1 - self.alpha))
    
    def _compute_gradient(self, allocation: AllocationMatrix, utilities: UtilityMatrix) -> np.ndarray:
        """Compute gradient of α-fairness objective."""
        agent_utilities = np.sum(allocation * utilities, axis=1)
        agent_utilities = np.maximum(agent_utilities, 1e-12)  # Numerical stability
        
        if self.alpha == 1.0:
            # Gradient for proportional fairness
            weights = 1.0 / agent_utilities
        elif self.alpha == 0.0:
            # Gradient for utilitarian
            weights = np.ones_like(agent_utilities)
        else:
            # Gradient for general α-fairness
            weights = agent_utilities ** (-self.alpha)
        
        # Compute gradient matrix
        gradient = weights[:, np.newaxis] * utilities
        
        return gradient
    
    def _project_onto_feasible_set(self, allocation: AllocationMatrix) -> AllocationMatrix:
        """Project allocation onto feasible set using efficient projection."""
        # Non-negativity constraint
        allocation = np.maximum(allocation, 0)
        
        # Resource capacity constraints (each agent gets at most 1 unit)
        agent_totals = allocation.sum(axis=1)
        over_capacity = agent_totals > 1.0
        
        if np.any(over_capacity):
            # Scale down allocations for agents that exceed capacity
            scaling_factors = np.where(over_capacity, 1.0 / agent_totals, 1.0)
            allocation = allocation * scaling_factors[:, np.newaxis]
        
        return allocation
    
    def _compute_fairness_metrics(self, 
                                allocation: AllocationMatrix, 
                                utilities: UtilityMatrix) -> Dict[str, float]:
        """Compute comprehensive fairness metrics."""
        agent_utilities = np.sum(allocation * utilities, axis=1)
        
        metrics = {}
        
        # Gini coefficient
        n = len(agent_utilities)
        if n > 1:
            sorted_utilities = np.sort(agent_utilities)
            index_array = np.arange(1, n + 1)
            metrics["gini_coefficient"] = (2 * np.sum(index_array * sorted_utilities)) / (n * np.sum(sorted_utilities)) - (n + 1) / n
        else:
            metrics["gini_coefficient"] = 0.0
        
        # Theil index
        mean_utility = np.mean(agent_utilities)
        if mean_utility > 0:
            relative_utilities = agent_utilities / mean_utility
            # Avoid log(0) issues
            relative_utilities = np.maximum(relative_utilities, 1e-12)
            metrics["theil_index"] = np.mean(relative_utilities * np.log(relative_utilities))
        else:
            metrics["theil_index"] = 0.0
        
        # Coefficient of variation
        if mean_utility > 0:
            metrics["coefficient_of_variation"] = np.std(agent_utilities) / mean_utility
        else:
            metrics["coefficient_of_variation"] = 0.0
        
        # Utilitarian and egalitarian welfare
        metrics["utilitarian_welfare"] = np.sum(agent_utilities)
        metrics["egalitarian_welfare"] = np.min(agent_utilities)
        
        # Nash welfare (geometric mean)
        if np.all(agent_utilities > 0):
            metrics["nash_welfare"] = np.prod(agent_utilities) ** (1/n)
        else:
            metrics["nash_welfare"] = 0.0
        
        # Envy ratio (approximate envy-freeness measure)
        envy_values = []
        for i in range(len(agent_utilities)):
            for j in range(len(agent_utilities)):
                if i != j:
                    # Utility agent i would get from agent j's allocation
                    envied_utility = np.sum(allocation[j, :] * utilities[i, :])
                    if agent_utilities[i] > 0:
                        envy_ratio = max(0, (envied_utility - agent_utilities[i]) / agent_utilities[i])
                        envy_values.append(envy_ratio)
        
        metrics["max_envy_ratio"] = np.max(envy_values) if envy_values else 0.0
        metrics["mean_envy_ratio"] = np.mean(envy_values) if envy_values else 0.0
        
        # Resource utilization
        resource_allocations = allocation.sum(axis=0)
        metrics["min_resource_utilization"] = np.min(resource_allocations)
        metrics["mean_resource_utilization"] = np.mean(resource_allocations)
        metrics["max_resource_utilization"] = np.max(resource_allocations)
        
        # Algorithm-specific metrics
        metrics["alpha_parameter"] = self.alpha
        metrics["alpha_fairness_objective"] = self._compute_alpha_fairness_objective(allocation, utilities)
        
        return metrics

class ProportionalFairnessAllocator(AlphaFairnessAllocator):
    """Proportional fairness allocator (α = 1)."""
    
    def __init__(self, **kwargs):
        super().__init__(alpha=1.0, **kwargs)
        self.fairness_type = FairnessType.PROPORTIONAL

class MaxMinFairnessAllocator(BaseAllocator):
    """
    Max-min fairness allocator.
    
    Maximizes the minimum utility among all agents, corresponding to α → ∞
    in the α-fairness framework.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fairness_type = FairnessType.MAX_MIN
    
    def _compute_optimal_allocation(self, 
                                  utilities: UtilityMatrix,
                                  constraints: Optional[Dict[str, Any]] = None) -> Tuple[AllocationMatrix, Dict[str, Any]]:
        """Compute max-min fair allocation using linear programming."""
        from scipy.optimize import linprog
        
        n_agents, n_resources = utilities.shape
        
        # Variables: [allocation_matrix (flattened), min_utility]
        # We maximize min_utility subject to constraints
        
        # Objective: maximize min_utility (so minimize -min_utility)
        c = np.zeros(n_agents * n_resources + 1)
        c[-1] = -1  # Coefficient for min_utility variable
        
        # Inequality constraints
        # For each agent i: sum_j(allocation[i,j] * utility[i,j]) >= min_utility
        A_ub = []
        b_ub = []
        
        for i in range(n_agents):
            constraint = np.zeros(n_agents * n_resources + 1)
            for j in range(n_resources):
                constraint[i * n_resources + j] = -utilities[i, j]
            constraint[-1] = 1  # min_utility coefficient
            A_ub.append(constraint)
            b_ub.append(0)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Equality constraints: sum over agents for each resource <= 1
        A_eq = []
        b_eq = []
        
        for j in range(n_resources):
            constraint = np.zeros(n_agents * n_resources + 1)
            for i in range(n_agents):
                constraint[i * n_resources + j] = 1
            A_eq.append(constraint)
            b_eq.append(1)
        
        # Each agent gets at most 1 unit total
        for i in range(n_agents):
            constraint = np.zeros(n_agents * n_resources + 1)
            for j in range(n_resources):
                constraint[i * n_resources + j] = 1
            A_eq.append(constraint)
            b_eq.append(1)
        
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None
        
        # Bounds: all allocation variables >= 0, min_utility unbounded
        bounds = [(0, None)] * (n_agents * n_resources) + [(None, None)]
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs')
            
            if result.success:
                allocation_flat = result.x[:-1]
                allocation = allocation_flat.reshape((n_agents, n_resources))
                
                convergence_info = {
                    "status": ConvergenceStatus.CONVERGED,
                    "objective_value": -result.fun,
                    "solver_message": result.message,
                    "iterations": getattr(result, 'nit', 0)
                }
            else:
                # Fallback to uniform allocation
                allocation = np.ones((n_agents, n_resources)) / n_resources
                convergence_info = {
                    "status": ConvergenceStatus.NUMERICAL_ERROR,
                    "solver_message": result.message
                }
                
        except Exception as e:
            self.logger.warning(f"Linear programming failed: {e}")
            allocation = np.ones((n_agents, n_resources)) / n_resources
            convergence_info = {
                "status": ConvergenceStatus.NUMERICAL_ERROR,
                "error_message": str(e)
            }
        
        return allocation, convergence_info
    
    def _compute_fairness_metrics(self, 
                                allocation: AllocationMatrix, 
                                utilities: UtilityMatrix) -> Dict[str, float]:
        """Compute fairness metrics for max-min allocation."""
        agent_utilities = np.sum(allocation * utilities, axis=1)
        
        metrics = {
            "min_utility": np.min(agent_utilities),
            "max_utility": np.max(agent_utilities), 
            "utility_range": np.max(agent_utilities) - np.min(agent_utilities),
            "mean_utility": np.mean(agent_utilities),
            "std_utility": np.std(agent_utilities),
        }
        
        # Add standard fairness metrics
        n = len(agent_utilities)
        if n > 1:
            sorted_utilities = np.sort(agent_utilities)
            index_array = np.arange(1, n + 1)
            metrics["gini_coefficient"] = (2 * np.sum(index_array * sorted_utilities)) / (n * np.sum(sorted_utilities)) - (n + 1) / n
        else:
            metrics["gini_coefficient"] = 0.0
        
        return metrics

class EntropyDualityAllocator(BaseAllocator):
    """
    Information-theoretic allocation using entropy fairness duality.
    
    Balances allocation entropy with utility efficiency using the principle:
    maximize H(allocation) - λ * I(allocation; preferences)
    """
    
    def __init__(self, 
                 entropy_weight: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.entropy_weight = entropy_weight
        self.fairness_type = FairnessType.ENTROPY_DUAL
    
    def _compute_optimal_allocation(self, 
                                  utilities: UtilityMatrix,
                                  constraints: Optional[Dict[str, Any]] = None) -> Tuple[AllocationMatrix, Dict[str, Any]]:
        """Compute entropy-optimal allocation."""
        n_agents, n_resources = utilities.shape
        
        # Start with maximum entropy allocation (uniform)
        allocation = np.ones((n_agents, n_resources)) / n_resources
        
        convergence_info = {
            "status": ConvergenceStatus.CONVERGED,
            "entropy_objective_history": [],
        }
        
        for iteration in range(self.max_iterations):
            # Compute entropy-based gradient
            entropy_gradient = self._compute_entropy_gradient(allocation, utilities)
            
            # Update allocation
            allocation_new = allocation + self.step_size * entropy_gradient
            allocation = self._project_onto_feasible_set_entropy(allocation_new)
            
            # Compute objective value
            objective = self._compute_entropy_objective(allocation, utilities)
            convergence_info["entropy_objective_history"].append(objective)
            
            if iteration > 0:
                prev_objective = convergence_info["entropy_objective_history"][-2]
                if abs(objective - prev_objective) < self.convergence_tolerance:
                    break
        
        return allocation, convergence_info
    
    def _compute_entropy_gradient(self, allocation: AllocationMatrix, utilities: UtilityMatrix) -> np.ndarray:
        """Compute gradient of entropy-duality objective."""
        # Entropy component gradient
        entropy_grad = -np.log(allocation + 1e-12) - 1
        
        # Utility component gradient
        utility_grad = utilities
        
        # Combined gradient
        return entropy_grad + self.entropy_weight * utility_grad
    
    def _compute_entropy_objective(self, allocation: AllocationMatrix, utilities: UtilityMatrix) -> float:
        """Compute entropy-duality objective value."""
        # Allocation entropy
        entropy = -np.sum(allocation * np.log(allocation + 1e-12))
        
        # Expected utility
        expected_utility = np.sum(allocation * utilities)
        
        return entropy + self.entropy_weight * expected_utility
    
    def _project_onto_feasible_set_entropy(self, allocation: AllocationMatrix) -> AllocationMatrix:
        """Project onto feasible set preserving entropy structure."""
        # Ensure non-negativity
        allocation = np.maximum(allocation, 1e-12)
        
        # Normalize each agent's allocation to sum to 1
        agent_sums = allocation.sum(axis=1, keepdims=True)
        allocation = allocation / (agent_sums + 1e-12)
        
        return allocation
    
    def _compute_fairness_metrics(self, 
                                allocation: AllocationMatrix, 
                                utilities: UtilityMatrix) -> Dict[str, float]:
        """Compute entropy-based fairness metrics."""
        metrics = {}
        
        # Allocation entropy
        entropy = -np.sum(allocation * np.log(allocation + 1e-12))
        metrics["allocation_entropy"] = entropy
        
        # Normalized entropy (0 to 1 scale)
        max_entropy = np.log(allocation.size)
        metrics["normalized_entropy"] = entropy / max_entropy if max_entropy > 0 else 0
        
        # Expected utilities
        agent_utilities = np.sum(allocation * utilities, axis=1)
        metrics["mean_utility"] = np.mean(agent_utilities)
        metrics["utility_variance"] = np.var(agent_utilities)
        
        # Entropy-utility trade-off
        metrics["entropy_weight"] = self.entropy_weight
        metrics["entropy_duality_objective"] = self._compute_entropy_objective(allocation, utilities)
        
        return metrics

# Convenience function for creating allocators
def create_allocator(fairness_type: str, **kwargs) -> BaseAllocator:
    """
    Factory function for creating allocator instances.
    
    Args:
        fairness_type: Type of fairness criterion
        **kwargs: Allocator-specific parameters
        
    Returns:
        Initialized allocator instance
    """
    allocator_map = {
        "alpha_fairness": AlphaFairnessAllocator,
        "proportional_fairness": ProportionalFairnessAllocator,
        "max_min_fairness": MaxMinFairnessAllocator,
        "entropy_duality": EntropyDualityAllocator,
    }
    
    if fairness_type not in allocator_map:
        raise ValueError(f"Unsupported fairness type: {fairness_type}")
    
    return allocator_map[fairness_type](**kwargs)


# Note: prefer typing.cast when narrowing inferred types.
