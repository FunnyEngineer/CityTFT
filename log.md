#### Abreviation for the result file
- CM: None
- DL: internal Illuminance Near Window
- ET: None
- Intertia: zones Thermal Inertia and warmUp Time
- LW: no idea
- SW: Irradiance
- SWv: illuminance (lux)
- TH: 
    - Heating (Wh): the ideal heating needs to reach the heating setpoint temperature,
    - Cooling (Wh): the ideal cooling needs to reach the cooling setpoint temperature,
    - Qi (Wh): the internal gains comprising occupants, appliances and solar heat gains,
    - Qs (Wh): the satisfied heating (positive) or cooling (negative) needs - this is the column to take into account for ideal loads calculation,
    - VdotVent (mᶟ/h): the natural ventilation flow rate by window openings,
    - HeatStockTemperature (°C): the temperature of the heat tank,
    - DHWStockTemperature (°C): the temperature of the domestic hot water tank (if applicable),
    - ColdStockTemperature (°C): the temperature of the cold tank,
    - MachinePower (W): the power necessary for the energy conversion system to provide heating or cooling,
    - FuelConsumption (MJ): the energy consumed by the energy conversion system in terms of fuel,
    - ElectricConsumption (kWh): the energy consumed by the energy conversion system in terms of electricity deduced by the photovoltaics (PV) production,
    - SolarThermalProduction (J): the energy provided to the heat stock by solar thermal panels.
- TS:
    - Ke(W/(m2K)) 
    - Tos(°C) sea surface temperatures
- YearlyResultsPerBuilding
    - heatingNeeds(Wh)	
    - coolingNeeds(Wh)


### example command to excute main

source main.sh -x ./data/SRLOD3.1_Annual_results.xml -c data/climate/AUS_cli/CAMP_MABRY_TX-contemporary.cli -p ./new_cli -n ./new_xml -e ./export
source main.sh -x ./data/SRLOD3.1_Annual_results.xml -c data/climate/AUS_cli/CAMP_MABRY_TX-RCP2.6-2030.cli -p ./new_cli -n ./new_xml -e ./export
source main.sh -x ./data/SRLOD3.1_Annual_results.xml -c data/climate/AUS_cli/CAMP_MABRY_TX-RCP2.6-2040.cli -p ./new_cli -n ./new_xml -e ./export
source main.sh -x ./data/SRLOD3.1_Annual_results.xml -c data/climate/AUS_cli/CAMP_MABRY_TX-RCP2.6-2050.cli -p ./new_cli -n ./new_xml -e ./export
source main.sh -x ./data/SRLOD3.1_Annual_results.xml -c data/climate/AUS_cli/CAMP_MABRY_TX-RCP4.5-2030.cli -p ./new_cli -n ./new_xml -e ./export
source main.sh -x ./data/SRLOD3.1_Annual_results.xml -c data/climate/AUS_cli/CAMP_MABRY_TX-RCP4.5-2040.cli -p ./new_cli -n ./new_xml -e ./export
source main.sh -x ./data/SRLOD3.1_Annual_results.xml -c data/climate/AUS_cli/CAMP_MABRY_TX-RCP4.5-2050.cli -p ./new_cli -n ./new_xml -e ./export
source main.sh -x ./data/SRLOD3.1_Annual_results.xml -c data/climate/AUS_cli/CAMP_MABRY_TX-RCP8.5-2030.cli -p ./new_cli -n ./new_xml -e ./export
source main.sh -x ./data/SRLOD3.1_Annual_results.xml -c data/climate/AUS_cli/CAMP_MABRY_TX-RCP8.5-2040.cli -p ./new_cli -n ./new_xml -e ./export
source main.sh -x ./data/SRLOD3.1_Annual_results.xml -c data/climate/AUS_cli/CAMP_MABRY_TX-RCP8.5-2050.cli -p ./new_cli -n ./new_xml -e ./export
source main.sh -x ./data/SRLOD3.1_Annual_results.xml -c data/climate/AUS_cli/CAMP_MABRY_TX-RCP8.5-2100.cli -p ./new_cli -n ./new_xml -e ./export
