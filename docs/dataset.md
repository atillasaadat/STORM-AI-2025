# Dataset Information

<!-- Announcement Quote Block -->
<!-- Announcement Quote Block -->
<div style="display: flex; flex-direction: column; background-color: #e7f3fe; border-left: 6px solid #2196f3; border-radius: 4px; padding: 15px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); font-family: Arial, sans-serif; color: #333;">
    <div style="font-size: 18px; font-weight: bold; color: #ffffff; background-color: #1565c0; display: inline-block; padding: 5px 10px; border-radius: 3px; margin-bottom: 10px;">📢 Announcement</div>
    <p style="font-size: 14px; margin: 0; line-height: 1.5;">
        The public challenge problem dataset is now available to participants with the start of <strong>Phase 1</strong> of the competition.
    </p>
</div>

## Challenge Dataset Availability

The public challenge dataset will be available for participants to download and use to train and develop their models during Phase 1 of the competition. The warmup dataset is a portion of the full public dataset and is representative of the formats participants can expect for all provided training data.

The private evaluation dataset contains data from spacecraft and/or time periods that are not represented in the public training data. It will retain the same format as the public dataset and will be used to evaluate the performance of the participants' models for the public leaderboard and final rankings.

## STORM-AI Data Formats

### Warmup Dataset 

The <a href="https://www.dropbox.com/scl/fo/nz1j92xpr6eet3fa5mx5i/ADMYs2zfr3dvxJ-FFd5dmM8?rlkey=tem27v1d2raf2nnlcq6cd49ev&st=sty6ggo3&dl=0)">Warm-Up Dataset</a> for participants was released on November 15, 2024. 

Participants should use this subset of the STORM-AI data to become familiar with the phenomenology and formats they will be asked to use during Phase 1 of the competition. The warmup dataset consists of the following files and folders: 
* <b>wu001_to_wu715-initial_states.csv</b>: 715 sets of orbital elements and geodetic position coordinates for ESA's SWARM A satellite
* <b>OMNI2</b>: Space weather information collected by NASA Space Flight Goddard Center and provided in 60-day segments (one 60-day OMNI2 history file per initial SWARM A state)
* <b>GOES</b>: X-Ray flux information collected by NOAA'S GOES-13 satellite and provided in 60-day segments (one 60-day GOES history file per initial SWARM A state)
* <b>Sat_Density</b>: Time series orbit average density values collected by ESA'S SWARM A satellite and provided in 3-day segments (one 3-day "forecasted" density file per initial SWARM A state)

Your objective is to design a model that, given a spacecraft's initial state and 60 days of space weather information directly preceding that state, can predict the next 3 days of atmospheric density values the spacecraft will observe.

That is, your model should take in these inputs: 
* A satellite's initial location, provided in both geodetic coordinates and orbital elements
* Space weather information for the 60 day period preceding the timestamp of the initial satellite location
* X-Ray flux information for the 60 day period preceding the timestamp of the initial satellite location

Your model should then predict the sequence of orbit-averaged atmospheric density values that the spacecraft will observe in the future. This prediction must span a period of 3 days directly following the timestamp of the initial satellite location.  

Some recommendations: 
* We recommend reviewing the data column header descriptions below for more details on the challenge and warmup dataset. This includes exploring the column headers and data type descriptions. 
* Be mindful of dataset units and representations! The warmup data should help get you into this mindset.
* You may access additional GOES, OMNI2, and SWARM A data through NOAA, NASA, and ESA online if you wish to supplement the provided warm-up data while training your algorithms.

### Definitions

#### Satellite Initial State Data
| Column Header  | Description | 
| ------------- | ------------- | 
| File ID  | 5 character designator assigned to the OMNI2, GOES, & Sat_Density files associated with the initial satellite state |
| Timestamp  | Start of the study period  |
| Semi-major Axis (km) | Satellite semi-major axis. Defines the size of the orbit. Represented in km | 
| Eccentricity | Satellite eccentricity. Describes the shape of the orbit and how stretched it is. Ranges from 0 (perfectly circular) to 1 (parabolic); values greater than 1 indicate a hyperbolic orbit. | 
| Inclination (deg) | Satellite inclination. The tilt of the orbit relative to the equatorial plane of the central body; the angle between the orbital plane and the equatorial plane. An inclination of 0° indicates an orbit in the equatorial plane, while 90° represents a polar orbit (crossing over the poles). Units in degrees. |
| RAAN (deg) | Satellite right ascension of the ascending node. Defines horizontal orientation of the orbit; tilt of the orbit relative to the equatorial plane. Units in degrees. |
| Argument of Perigee (deg) | Satellite argument of periapsis. Specifies the orientation of the orbit within its plane; defines where the orbit’s closest approach occurs within the plane of the orbit. Units in degrees. |
| True Anomaly (deg) | Satellite true anomaly. Position of the body along the orbit at a specific time. Units in degrees. |
| Latitude (deg) | Satellite geodetic latitude at the start of the study period. Units in degrees.   | 
| Longitude (deg) | Satellite geodetic longitude at the start of the study period. Units in degrees. |
| Altitude (km) | Satellite altitude at the start of the study period. Units in km for this dataset.  | 

#### OMNI2 Space Weather Data
| Column Header              | Description                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------|
| Timestamp                  | Datetime-like timestamp of observation.                                                               |
| YEAR                       | Year of observation.                                                                                 |
| DOY                        | Day of Year (Julian day) of observation. Ranges 01 to 365.                                           |
| Hour                       | Hour of observation, typically in UTC. Ranges 00 to 23.                                              |
| Bartels_rotation_number    | Number of 27-day solar rotations that have occurred since February 8, 1832.                        |
| ID_for_IMF_spacecraft      | Identifier for spacecraft measuring IMF (Interplanetary Magnetic Field).                  |
| ID_for_SW_Plasma_spacecraft| Identifier for spacecraft measuring Solar Wind Plasma.                                   |
| num_points_IMF_averages    | Number of data points used to calculate IMF averages.                                   |
| num_points_Plasma_averages | Number of data points used to calculate Plasma averages.                               |
| Scalar_B_nT                | Scalar (magnitude) of the magnetic field. Units in nT.                                               |
| Vector_B_Magnitude_nT      | Vector magnitude of the magnetic field. Units in nT.                                                 |
| Lat_Angle_of_B_GSE         | Latitude angle of magnetic field in GSE coordinates. Units in Degrees.                               |
| Long_Angle_of_B_GSE        | Longitude angle of magnetic field in GSE coordinates. Units in Degrees.                              |
| BX_nT_GSE_GSM              | X-component of magnetic field in GSE and GSM coordinates. Units in nT.                               |
| BY_nT_GSE                  | Y-component of magnetic field in GSE coordinates. Units in nT.                                       |
| BZ_nT_GSE                  | Z-component of magnetic field in GSE coordinates. Units in nT.                                       |
| BY_nT_GSM                  | Y-component of magnetic field in GSM coordinates. Units in nT.                                       |
| BZ_nT_GSM                  | Z-component of magnetic field in GSM coordinates. Units in nT.                                       |
| RMS_magnitude_nT           | RMS of magnetic field magnitude. Units in nT.                                                        |
| RMS_field_vector_nT        | RMS of the magnetic field vector. Units in nT.                                                       |
| RMS_BX_GSE_nT              | RMS of BX component in GSE. Units in nT.                                                             |
| RMS_BY_GSE_nT              | RMS of BY component in GSE. Units in nT.                                                             |
| RMS_BZ_GSE_nT              | RMS of BZ component in GSE. Units in nT.                                                             |
| SW_Plasma_Temperature_K    | Solar wind plasma temperature. Units in K.                                                           |
| SW_Proton_Density_N_cm3    | Solar wind proton density. Units in particles/cm³.                                                   |
| SW_Plasma_Speed_km_s       | Solar wind plasma speed. Units in km/s.                                                              |
| SW_Plasma_flow_long_angle  | Longitude angle of solar wind plasma flow. Units in Degrees.                                         |
| SW_Plasma_flow_lat_angle   | Latitude angle of solar wind plasma flow. Units in Degrees.                                          |
| Alpha_Prot_ratio           | Ratio of alpha particles to protons in the solar wind. Units in -.                                   |
| sigma_T_K                  | Standard deviation of solar wind temperature. Units in K.                                            |
| sigma_n_N_cm3              | Standard deviation of proton density. Units in particles/cm³.                                        |
| sigma_V_km_s               | Standard deviation of solar wind speed. Units in km/s.                                               |
| sigma_phi_V_degrees        | Standard deviation of plasma flow longitude angle. Units in Degrees.                                 |
| sigma_theta_V_degrees      | Standard deviation of plasma flow latitude angle. Units in Degrees.                                  |
| sigma_ratio                | Standard deviation ratio. Units in -.                                                                |
| Flow_pressure              | Solar wind flow pressure. Units in nPa.                                                              |
| E_electric_field           | Electric field strength. Units in mV/m.                                                              |
| Plasma_Beta                | Ratio of plasma pressure to magnetic pressure. Units in -.                                           |
| Alfen_mach_number          | Alfvén Mach Number, indicating ratio of solar wind speed to Alfvén speed. Units in -.                |
| Magnetosonic_Mach_number   | Magnetosonic Mach Number, indicating ratio of solar wind speed to magnetosonic speed. Units in -.    |
| Quasy_Invariant            | Quasi-invariant parameter in plasma physics. Units in -.                                             |
| Kp_index                   | Planetary K-index, measuring geomagnetic activity. Units in -.                                       |
| R_Sunspot_No               | Sunspot number, indicating solar activity level. Units in -.                                         |
| Dst_index_nT               | Disturbance Storm Time index. Units in nT.                                                           |
| ap_index_nT                | Ap index, daily averaged planetary geomagnetic index. Units in nT.                                   |
| f10.7_index                | F10.7 Solar radio flux. Units in 10⁻²² W/m²/Hz.                                                      |
| AE_index_nT                | Auroral Electrojet Index, measuring auroral activity strength. Units in nT.                          |
| AL_index_nT                | Auroral Lower Index, measuring lower limit of auroral activity. Units in nT.                         |
| AU_index_nT                | Auroral Upper Index, measuring upper limit of auroral activity. Units in nT.                         |
| pc_index                   | Polar Cap Index, measuring geomagnetic activity in polar regions. Units in -.                        |
| Lyman_alpha                | Lyman-alpha solar flux. Units in W/m².                                                               |
| Proton_flux_>1_Mev         | Proton flux for energies >1 MeV. Units in particles/cm²/s/sr.                                        |
| Proton_flux_>2_Mev         | Proton flux for energies >2 MeV. Units in particles/cm²/s/sr.                                        |
| Proton_flux_>4_Mev         | Proton flux for energies >4 MeV. Units in particles/cm²/s/sr.                                        |
| Proton_flux_>10_Mev        | Proton flux for energies >10 MeV. Units in particles/cm²/s/sr.                                       |
| Proton_flux_>30_Mev        | Proton flux for energies >30 MeV. Units in particles/cm²/s/sr.                                       |
| Proton_flux_>60_Mev        | Proton flux for energies >60 MeV. Units in particles/cm²/s/sr.                                       |
| Flux_FLAG                  | Flag indicating quality of proton flux measurements. Units in -.                                     |
| Date                       | Date of observation. Units in YYYY-MM-DD.                                                            |

#### GOES-13 X-Ray Flux Data
| Column Header            | Description                                                                                       |
|---------------------------|---------------------------------------------------------------------------------------------------|
| Timestamp                 | Timestamp of the measurement.    
| xrsa_flux                | Flux measured in the XRS-A sensor, represents solar soft X-ray emissions. |
| xrsa_flux_observed       | Observed flux in the XRS-A sensor, including all raw measurement data without corrections.         |
| xrsa_flux_electrons      | Estimated contribution of electron flux in the XRS-A sensor measurements.       |
| xrsb_flux                | Flux measured in the XRS-B sensor, represents solar hard X-ray emissions. |
| xrsb_flux_observed       | Observed flux in the XRS-B sensor, including all raw measurement data without corrections.         |
| xrsb_flux_electrons      | Estimated contribution of electron flux in the XRS-B sensor measurements.      |
| xrsa_flag                | Quality flag for XRS-A data. Binary indicator: 0 for valid data, 1 for flagged or questionable data. |
| xrsb_flag                | Quality flag for XRS-B data. Binary indicator: 0 for valid data, 1 for flagged or questionable data. |
| xrsa_num                 | Number of valid data points in the XRS-A dataset during the study period.                          |
| xrsb_num                 | Number of valid data points in the XRS-B dataset during the study period.                          |
| xrsa_flag_excluded       | Indicates whether specific XRS-A data points are excluded based on quality checks. Binary flag.    |
| xrsb_flag_excluded       | Indicates whether specific XRS-B data points are excluded based on quality checks. Binary flag.    |

#### SWARM A Atmospheric Density Data
| Column Header  | Description | 
| ------------- | ------------- | 
| Timestamp  | Datetime-like timestamp of observation. |
| Orbit Mean Density (kg/m^3) | Orbit-average of density derived from GPS accelerations. Helps assess long-term drag effects; used to decide if drift control maneuvers are needed due to accumulated orbital decay. Units in kg/m^3. | 

### Release History

| Version  | Release Date    | Description                                                                                           |
|--------- | --------------- | ----------------------------------------------------------------------------------------------------- |
| V1.0     | 2024-11-15      | OMNI2 and SWARM A data shared in 3 files (OMNI2, SWARM A POD, SWARM A DNS) for 2014-2019              |
| V1.1     | 2024-11-20      | Added GOES East inputs for 2014-2019                                                                  |
| V2.0     | 2024-12-11      | Reorganized OMNI2, GOES & SWARM data into multiple files (1 of each type per initial satellite state) |

## Guidelines

Participants should note that algorithm inputs must be limited to the phenomenology and data formats present in the public training dataset, but utilizing additional phenomenology or data sources for model validation and development is allowed and encouraged. 

Also note that since these datasets are collected from real spacecraft, there may be short gaps in the data due to blackout periods and instrument error. Participants should take this into account when designing their algorithms to ensure successful submissions.

