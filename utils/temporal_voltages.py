import numpy as np


class TemporalVoltage:
    """
    Represents a time-dependent voltage applied to a specific node in a mesh.

    Attributes
    ----------
    node_index : int
        The index of the node where the voltage is applied.
    time_sequence : np.ndarray
        A 1D numpy array of voltage values between -1 and 1.
    """
    def __init__(self, node_index: int, time_sequence: np.ndarray):
        """
        Initializes the TemporalVoltage object.

        Parameters
        ----------
        node_index : int
            The index of the node where the voltage is applied.
        time_sequence : np.ndarray
            A 1D numpy array of voltage values.

        Raises
        ------
        TypeError
            If node_index is not an integer or if time_sequence is not a numpy array.
        ValueError
            If time_sequence is not a 1D array or if its values are not between -1 and 1.
        """
        if not isinstance(node_index, int):
            raise TypeError("node_index must be an integer.")
        if not isinstance(time_sequence, np.ndarray):
            raise TypeError("time_sequence must be a numpy array.")
        if time_sequence.ndim != 1:
            raise ValueError("time_sequence must be a 1D array.")


        self.node_index = node_index
        self.time_sequence = time_sequence


class SineVoltage(TemporalVoltage):
    """
    Represents a sinusoidal time-dependent voltage applied to a specific node.

    This class generates the time sequence for a sine wave based on given
    parameters and initializes the parent TemporalVoltage class with it.
    """
    def __init__(self, node_index: int, period_length: float, time_length: int, amplitude: float, offset: int = 0):
        """
        Initializes the SineVoltage object.

        Parameters
        ----------
        node_index : int
            The index of the node where the voltage is applied.
        period_length : float
            The number of time steps in one period of the sine wave.
        time_length : int
            The total number of time steps for the voltage sequence.
        amplitude : float
            The amplitude of the sine wave. The generated values will be clipped
            to the range [-1, 1] as required by the parent class.
        offset : int, optional
            The number of time steps to wait with zero voltage before the
            sine wave begins (default is 0).
        """
        if not isinstance(period_length, (int, float)) or period_length <= 0:
            raise ValueError("period_length must be a positive number.")
        if not isinstance(time_length, int) or time_length <= 0:
            raise ValueError("time_length must be a positive integer.")
        if not isinstance(amplitude, (int, float)):
            raise TypeError("amplitude must be a number.")
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer.")
        if offset >= time_length:
            raise ValueError("offset must be less than time_length.")

        wave_duration = time_length - offset
        time_points = np.arange(wave_duration)
        sine_wave = amplitude * np.sin(2 * np.pi * time_points / period_length)

        time_sequence = np.zeros(time_length)
        if wave_duration > 0:
            time_sequence[offset:] = sine_wave

        super().__init__(node_index, time_sequence)


class NPhasesVoltage(TemporalVoltage):
    """
    Represents a multi-phase time-dependent voltage applied to a specific node.
    
    This class generates a time sequence where different voltage values are applied
    for different phases of the total duration. The phases are distributed as evenly
    as possible, with any remainder time added to the last phase.
    
    Attributes
    ----------
    node_index : int
        The index of the node where the voltage is applied.
    voltage_values : np.ndarray
        Array of voltage values for each phase.
    duration : int
        Total duration (number of time steps).
    """
    
    def __init__(self, node_index: int, voltage_values: np.ndarray, duration: int):
        self.voltage_values = voltage_values
        self.duration = duration
        
        # Generate the time sequence
        time_sequence = self._generate_phase_sequence()
        
        # Initialize parent class
        super().__init__(node_index, time_sequence)
    
    def _generate_phase_sequence(self) -> np.ndarray:
        """
        Generates the phase-based voltage sequence.
        
        Returns
        -------
        np.ndarray
            The generated time sequence with phase-based voltages.
        """
        n_phases = len(self.voltage_values)
        
        # Calculate phase durations
        base_duration = self.duration // n_phases
        remainder = self.duration % n_phases
        
        # Create the time sequence
        time_sequence = np.zeros(self.duration)
        
        current_index = 0
        for i, voltage in enumerate(self.voltage_values):
            # Calculate duration for this phase
            if i == n_phases - 1:  # Last phase gets the remainder
                phase_duration = base_duration + remainder
            else:
                phase_duration = base_duration
            
            # Set voltage for this phase
            end_index = current_index + phase_duration
            time_sequence[current_index:end_index] = voltage
            current_index = end_index
        
        return time_sequence
