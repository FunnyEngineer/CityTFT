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


### 2023.08.30
Experimental Evaluation
1. Number in Test Dataset

| MODEL | # of data |
| CityDNN | 123 |
| CityRNN | 199744 |

| MODEL | COOL MSE | HEAT MSE | TOTAL MSE |
| CityDNN | 850754535.3309836 | 711284811.7065043 | 1562039347.037488 |
| CityRNN | 30026729.56328301 | 34863257.79955027 | 64889987.36283328 |


## 2023.09.05 Discussion
1. increase the weather file in dataset

2. the defintion -> output file
Suggorate model -> sequence modeling


## 2023.09.07 Config setting
1. Starting to develop TFT model
    Code taken from "https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Forecasting/TFT"

2. Starting setting config in their repo
    -> building props = STATIC


# model size comparision
 - RNN Sequence Model

  | Name             | Type       | Params
------------------------------------------------
0 | encoder          | LSTM       | 816 K 
1 | heat_decoder     | Sequential | 33.3 K
2 | cool_decoder     | Sequential | 33.3 K
3 | heat_decoder_seq | LSTM       | 1.1 M 
4 | cool_decoder_seq | LSTM       | 1.1 M 
------------------------------------------------
3.0 M     Trainable params
0         Non-trainable params
3.0 M     Total params
11.952    Total estimated model params size (MB)


 - Hybrid Trans-RNN model
  | Name             | Type               | Params
--------------------------------------------------------
0 | encoder          | TransformerEncoder | 792 K 
1 | heat_decoder     | Sequential         | 33.3 K
2 | cool_decoder     | Sequential         | 33.3 K
3 | heat_decoder_seq | LSTM               | 1.1 M 
4 | cool_decoder_seq | LSTM               | 1.1 M 
5 | linear_embed     | Linear             | 6.7 K 
--------------------------------------------------------

 - Transformer Model (With TS involved)
  | Name             | Type               | Params
--------------------------------------------------------
0 | encoder          | TransformerEncoder | 792 K 
1 | heat_decoder     | Sequential         | 33.3 K
2 | cool_decoder     | Sequential         | 33.3 K
3 | heat_decoder_seq | TransformerDecoder | 1.3 M 
4 | cool_decoder_seq | TransformerDecoder | 1.3 M 
5 | linear_embed     | Linear             | 6.7 K 
6 | linear_embed_ts  | Linear             | 512   
--------------------------------------------------------
3.5 M     Trainable params
0         Non-trainable params
3.5 M     Total params
14.017    Total estimated model params size (MB)

- Transformer Model V2 (Encoder for decoder)
  | Name             | Type               | Params
--------------------------------------------------------
0 | encoder          | TransformerEncoder | 792 K 
1 | heat_decoder     | Sequential         | 33.3 K
2 | cool_decoder     | Sequential         | 33.3 K
3 | heat_decoder_seq | TransformerEncoder | 792 K 
4 | cool_decoder_seq | TransformerEncoder | 792 K 
5 | linear_embed     | Linear             | 6.9 K 
6 | linear_embed_ts  | Linear             | 512   
--------------------------------------------------------
2.4 M     Trainable params
0         Non-trainable params
2.4 M     Total params
9.797     Total estimated model params size (MB)
Epoch 0:   0%|                                    

