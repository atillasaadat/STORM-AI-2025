import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pymor.basic import *
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
# Initialize Orekit and JVM
import orekit
from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.frames import FramesFactory, LOFType, Frame
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.propagation import SpacecraftState
from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.models.earth.atmosphere import JB2008, PythonAtmosphere
from org.orekit.forces.drag import IsotropicDrag, DragForce
from org.orekit.utils import IERSConventions
from org.hipparchus.geometry.euclidean.threed import Vector3D

import matplotlib.pyplot as plt

# Initialize Orekit
orekit.initVM()
setup_orekit_curdir()

# =============================================================================
# Define Constants and Initial Conditions
# =============================================================================
r_Earth = Constants.IERS2010_EARTH_EQUATORIAL_RADIUS  # Earth radius (m)
mu = Constants.IERS2010_EARTH_MU  # Earth's gravitational parameter (m^3/s^2)
inertialFrame = FramesFactory.getEME2000()
utc = TimeScalesFactory.getUTC()
degree, torder = 70, 70  # Gravity field model
step, horizon = 600.0, 86400.0  # Propagation step: 10 minutes, 1-day horizon
satellite_mass, cross_section, drag_coeff = 260.0, 3.2 * 1.6, 2.2

class AtmosphericDensityNN(nn.Module):
    """Neural network model for atmospheric density prediction."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AtmosphericDensityNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AtmosphericDensityEstimator:
    """Real-time atmospheric density estimation using a neural network."""
    def __init__(self, training_data, input_data):
        """
        Initialize the NN-based density estimator.
        :param training_data: Atmospheric density snapshots.
        :param input_data: External inputs affecting the atmosphere.
        """
        training_data = np.log10(np.array(training_data) + 1e-12)  # Log-density representation
        input_data = np.array(input_data)
        
        self.model = AtmosphericDensityNN(input_dim=2, hidden_dim=16, output_dim=1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.train_model(input_data, training_data)
    
    def train_model(self, inputs, targets):
        """Train the neural network on historical data."""
        inputs, targets = torch.Tensor(inputs), torch.Tensor(targets)
        for epoch in range(1000):
            self.optimizer.zero_grad()
            predictions = self.model(inputs).squeeze()
            loss = self.criterion(predictions, targets)
            loss.backward()
            self.optimizer.step()
    
    def update_density(self, measurement):
        """Predicts the atmospheric density given input parameters."""
        measurement = torch.Tensor([[measurement, 0]])  # Ensure it has shape (1,2)
        return 10 ** self.model(measurement).item()


from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.utils import PVCoordinates

class CustomAtmosphere(PythonAtmosphere):
    """Neural network-based atmosphere model."""
    def __init__(self, density_estimator):
        super().__init__()
        self.density_estimator = density_estimator
        self.earth = OneAxisEllipsoid(
            Constants.IERS2010_EARTH_EQUATORIAL_RADIUS,
            Constants.IERS2010_EARTH_FLATTENING,
            FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        )

    def getDensity(self, date, position, frame):
        """Returns the estimated atmospheric density."""
        return self.density_estimator.update_density(0)  # Uses previous step's prediction

    def getVelocity(self, date, position, frame):
        """
        Returns the velocity of atmospheric molecules assuming Earth's rotation.
        """
        # Get the transform from the Earth's body frame to the inertial frame
        body_to_inertial = self.earth.getBodyFrame().getTransformTo(frame, date)
        
        # Transform position to the Earth's body frame
        pos_in_body = body_to_inertial.getInverse().transformPosition(position)
        
        # Compute velocity in Earth's body frame (assume rotational velocity)
        omega_earth = Constants.WGS84_EARTH_ANGULAR_VELOCITY  # rad/s
        velocity_body = Vector3D.crossProduct(Vector3D(0, 0, omega_earth), pos_in_body)
        
        # Transform velocity back to the inertial frame
        pv_body = PVCoordinates(pos_in_body, velocity_body)
        pv_inertial = body_to_inertial.transformPVCoordinates(pv_body)
        
        return pv_inertial.getVelocity()  # Return velocity in inertial frame
    
def train_density_model():
    """Generates synthetic density data for NN training."""
    densities = [1e-12 * (1 + 0.05 * np.sin(0.1 * t)) for t in range(100)]
    input_data = [[150 + 10 * np.sin(0.1 * t), 3 + np.cos(0.1 * t)] for t in range(100)]  # Example inputs
    return AtmosphericDensityEstimator(densities, input_data)

def prop_orbit(initial_orbit, duration, step):
    """Propagates the satellite orbit using NN-based atmosphere model."""
    satellite_mass = 260.0
    cross_section = 3.2 * 1.6  # m²
    drag_coeff = 2.2

    start_time = initial_orbit.getDate()
    tspan = [start_time.shiftedBy(float(dt)) for dt in np.linspace(0, duration, int(duration / step))]

    integrator = DormandPrince853Integrator(1e-6, 100.0, 1e-4, 1e-4)
    propagator = NumericalPropagator(integrator)
    propagator.setInitialState(SpacecraftState(initial_orbit, satellite_mass))

    density_estimator = train_density_model()
    atmosphere = CustomAtmosphere(density_estimator)
    dragForce = DragForce(atmosphere, IsotropicDrag(cross_section, drag_coeff))
    propagator.addForceModel(dragForce)

    states, densities, positions = [], [], []
    for t in tspan:
        state = propagator.propagate(t)
        pos = state.getPVCoordinates().getPosition()
        density = atmosphere.getDensity(t, pos, inertialFrame)
        states.append(state)
        densities.append(density)
        positions.append([pos.getX(), pos.getY(), pos.getZ()])

    positions = np.array(positions)
    plt.figure()
    plt.plot(positions[:, 0], positions[:, 1], label='Orbit Path')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Satellite Orbit Path')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(range(len(densities)), densities, label='Atmospheric Density')
    plt.xlabel('Time Step')
    plt.ylabel('Density (kg/m³)')
    plt.title('Density Time Series')
    plt.legend()
    plt.show()
    
    return states, densities

def __main__():
    a0 = float(r_Earth + 500e3)  # Semi-major axis (500km altitude)
    e0 = float(0.001)  # Eccentricity
    i0 = float(np.radians(51.6))  # Inclination (radians)
    w0 = float(np.radians(30))  # Argument of Perigee (radians)
    RAAN = float(np.radians(0))  # Right Ascension of Ascending Node (radians)
    M0 = float(np.radians(0))  # Mean Anomaly (radians)
    
    date = AbsoluteDate(2022, 1, 1, 0, 0, 0.0, TimeScalesFactory.getUTC())
    
    initialOrbit = KeplerianOrbit(a0, e0, i0, w0, RAAN, M0, PositionAngleType.TRUE, inertialFrame, date, mu)
    duration, step = 86400.0, 600.0  # 1 day, 10 min step
    states, densities = prop_orbit(initialOrbit, duration, step)

if __name__ == "__main__":
    __main__()
