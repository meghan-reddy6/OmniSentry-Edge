"""
PID Controller utility for servo feedback tracking loops.
"""
import time
import logging

logger = logging.getLogger(__name__)

class PIDController:
    """A standard Proportional-Integral-Derivative controller with anti-windup."""
    def __init__(
        self, 
        kp: float, 
        ki: float, 
        kd: float, 
        min_output: float = -20.0,  # Max rotation increment step per frame
        max_output: float = 20.0,
        integral_limit: float = 10.0
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.integral_limit = integral_limit
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def reset(self):
        """Resets the controller state (integral and previous error)."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def update(self, error: float, dt: float = None) -> float:
        """
        Calculates the PID control output.
        
        Args:
            error: The current system error (e.g. target_pos - current_pos).
            dt: Optional time delta. If None, it is calculated from the system clock.
            
        Returns:
            float: Control output bounded within [min_output, max_output].
        """
        now = time.time()
        if dt is None:
            if self.last_time is None:
                dt = 0.0
            else:
                dt = now - self.last_time
        
        self.last_time = now

        # Fallback if time did not advance
        if dt <= 0.0:
            # Revert to simple proportional control if time step is too small
            return self.kp * error

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup clamping
        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        i_term = self.ki * self.integral

        # Derivative term
        d_term = 0.0
        if dt > 0.0:
            derivative = (error - self.prev_error) / dt
            d_term = self.kd * derivative

        self.prev_error = error

        # Total control output
        output = p_term + i_term + d_term

        # Clamp output to limit peak velocity/change per update
        clamped_output = max(min(output, self.max_output), self.min_output)
        
        return clamped_output
