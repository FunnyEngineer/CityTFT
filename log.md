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

- Transformer Model V2 (Applied Encoder for decoder)
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



## 2023.09.16 new model architecture

- Transformer Model V2 (larget model architecture)
  | Name             | Type               | Params
--------------------------------------------------------
0 | encoder          | TransformerEncoder | 792 K 
1 | heat_decoder     | Sequential         | 33.3 K
2 | cool_decoder     | Sequential         | 33.3 K
3 | heat_decoder_seq | TransformerEncoder | 2.6 M 
4 | cool_decoder_seq | TransformerEncoder | 2.6 M 
5 | linear_embed     | Linear             | 6.9 K 
6 | linear_embed_ts  | Linear             | 512   
--------------------------------------------------------
6.1 M     Trainable params
0         Non-trainable params
6.1 M     Total params
24.505    Total estimated model params size (MB)

 - Sequence RNN Model (With Linear embed first)

  | Name             | Type       | Params
------------------------------------------------
0 | encoder          | LSTM       | 1.1 M 
1 | heat_decoder     | Sequential | 33.3 K
2 | cool_decoder     | Sequential | 33.3 K
3 | heat_decoder_seq | LSTM       | 1.1 M 
4 | cool_decoder_seq | LSTM       | 1.1 M 
5 | linear_embed     | Linear     | 6.7 K 
------------------------------------------------
3.2 M     Trainable params
0         Non-trainable params
3.2 M     Total params
12.925    Total estimated model params size (MB)


## 2023.09.19

- Climate zone definiation
Tropical (A): This category includes hot and humid climates found near the equator, such as tropical rainforests (Af) and tropical savannas (Aw).

Arid (B): Arid climates are characterized by low precipitation levels and include desert climates (BWh and BWk).

Temperate (C): Temperate climates have distinct seasons and are further subdivided into subcategories like humid subtropical (Cfa), Mediterranean (Csa and Csb), and marine west coast (Cfb and Cfc) climates.

Polar (D): Polar climates are cold year-round and include tundra (ET) and ice cap (EF) climates.

Highland (H): Highland or mountain climates vary depending on elevation and local geography.

- Starting multiploe climate file training

Training (14) + Validation (3) + Test (4) = Total climate zone

We have 13 C + 6 A + 2 B

Final Decide to
    |   Train   |  Val  |  Test  
------------------------------------------
C   |     9     |   2   |   2     
A   |     4     |   1   |   1     
B   |     1     |   0   |   1     


## 2023.09.21

- Finish first training for multiple climate zone
  | Name             | Type               | Params
--------------------------------------------------------
0 | encoder          | TransformerEncoder | 792 K 
1 | heat_decoder     | Sequential         | 33.3 K
2 | cool_decoder     | Sequential         | 33.3 K
3 | heat_decoder_seq | TransformerEncoder | 2.6 M 
4 | cool_decoder_seq | TransformerEncoder | 2.6 M 
5 | linear_embed     | Linear             | 6.9 K 
6 | linear_embed_ts  | Linear             | 512   
--------------------------------------------------------
6.1 M     Trainable params
0         Non-trainable params
6.1 M     Total params
24.505    Total estimated model params size (MB)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test/cool_loss       │    0.35428643226623535    │
│      test/heat_loss       │    0.10620741546154022    │
│      test/total_loss      │    0.46049439907073975    │
└───────────────────────────┴───────────────────────────┘

- CityRNN


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

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test/cool_loss       │    0.43139025568962097    │
│      test/heat_loss       │    0.09615669399499893    │
│      test/total_loss      │    0.5275468230247498     │
└───────────────────────────┴───────────────────────────┘

- CityTFT



## 2023.09.27 

-RNN

  | Name             | Type         | Params
--------------------------------------------------
0 | encoder          | LSTM         | 817 K 
1 | heat_decoder     | Sequential   | 33.5 K
2 | cool_decoder     | Sequential   | 33.5 K
3 | heat_decoder_seq | LSTM         | 1.1 M 
4 | cool_decoder_seq | LSTM         | 1.1 M 
5 | quantile_loss    | QuantileLoss | 0     
6 | heat_tigger      | Sequential   | 33.3 K
7 | cool_trigger     | Sequential   | 33.3 K
--------------------------------------------------
3.1 M     Trainable params
0         Non-trainable params
3.1 M     Total params
12.225    Total estimated model params size (MB)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        global_step        │         3641256.0         │
│      test/cool_loss       │    0.35578641295433044    │
│    test/cool_prob_loss    │     1.038880705833435     │
│      test/heat_loss       │    0.32541462779045105    │
│    test/heat_prob_loss    │    1.1125640869140625     │
│      test/total_loss      │     2.832646369934082     │
└───────────────────────────┴───────────────────────────┘

- Transformer
  | Name             | Type               | Params
--------------------------------------------------------
0 | encoder          | TransformerEncoder | 792 K 
1 | heat_decoder     | Sequential         | 33.5 K
2 | cool_decoder     | Sequential         | 33.5 K
3 | heat_decoder_seq | TransformerEncoder | 2.6 M 
4 | cool_decoder_seq | TransformerEncoder | 2.6 M 
5 | quantile_loss    | QuantileLoss       | 0     
6 | heat_tigger      | Sequential         | 33.3 K
7 | cool_trigger     | Sequential         | 33.3 K
8 | linear_embed     | Sequential         | 7.4 K 
--------------------------------------------------------
6.2 M     Trainable params
0         Non-trainable params
6.2 M     Total params
24.776    Total estimated model params size (MB)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        global_step        │         3641256.0         │
│      test/cool_loss       │    0.43877801299095154    │
│    test/cool_prob_loss    │    0.9994622468948364     │
│      test/heat_loss       │    0.23862263560295105    │
│    test/heat_prob_loss    │    0.8554173707962036     │
│      test/total_loss      │    2.5322790145874023     │
└───────────────────────────┴───────────────────────────┘

- TFT


## 0928 metric calculate
% there should be an F1 table, an overall loads difference table
% a table compare with only non-zero loads.
% there should be a figure compare actual loads and predicted loads.

### using last model
RNN F1 score: 0.9191 (heat: 0.9322, cool: 0.9010)
RNN RMSE: 75910.4453 (heat: 64946.2422, cool: 85479.6641)
RNN RMSE (non-zero): 114065.9453 (heat: 110553.9453, cool: 115777.4375)

Transformer F1 score: 0.9133 (heat: 0.9305, cool: 0.8896)
Transformer RMSE: 79739.5938 (heat: 61296.0508, cool: 94655.1719)
Transformer RMSE (non-zero): 118425.0938 (heat: 95192.5547, cool: 128449.5234)

TFT F1 score: 0.9998 (heat: 0.9999, cool: 0.9997)
TFT RMSE: 13571.3750 (heat: 9463.6895, cool: 16697.3965)
TFT RMSE (non-zero): 21342.7500 (heat: 18244.0820, cool: 22730.6309)


## using best model
RNN F1 score: 0.8963 (heat: 0.8993, cool: 0.8923)
RNN RMSE: 79832.0781 (heat: 67001.8906, cool: 90868.4141)
RNN RMSE (non-zero): 113541.0625 (heat: 91755.4766, cool: 122970.8281)
Transformer F1 score: 0.8694 (heat: 0.8791, cool: 0.8574)
Transformer RMSE: 103308.5469 (heat: 86652.0312, cool: 117629.6562)
Transformer RMSE (non-zero): 150222.5781 (heat: 130262.8672, cool: 159243.5625)

RNN MAPE: 1.3689 (heat: 2.8737, cool: 0.6186)
Transformer MAPE: 1.1365 (heat: 2.0588, cool: 0.6766)
TFT MAPE: 0.1162 (heat: 0.1292, cool: 0.1097)


#### log csv
, F1 score, non-zero RMSE, Total RMSE
RNN, 0.9191, 114065.9453, 75910.4453
Transformer, 0.9133, 118425.0938, 79739.5938
TFT, 0.9998, 21342.7500, 13571.3750
Heat Only, , ,
RNN, 0.9322, 110553.9453, 64946.2422
Transformer, 0.9305, 95192.5547, 61296.0508
TFT, 0.9999, 18244.0820, 9463.6895
Cool Only, , ,
RNN, 0.9010, 115777.4375, 85479.6641
Transformer, 0.8896, 128449.5234, 94655.1719
TFT, 0.9997, 22730.6309, 16697.3965


## 2023.10.26

Start to examine the TFT model.

## 2023.10.29

 - Training with less hidden dim
  | Name             | Type         | Params
--------------------------------------------------
0 | encoder          | LSTM         | 211 K 
1 | heat_decoder     | Sequential   | 8.6 K 
2 | cool_decoder     | Sequential   | 8.6 K 
3 | heat_decoder_seq | LSTM         | 264 K 
4 | cool_decoder_seq | LSTM         | 264 K 
5 | quantile_loss    | QuantileLoss | 0     
6 | heat_tigger      | Sequential   | 8.4 K 
7 | cool_trigger     | Sequential   | 8.4 K 
--------------------------------------------------
774 K     Trainable params
0         Non-trainable params
774 K     Total params
3.098     Total estimated model params size (MB)

 ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        global_step        │         3641256.0         │
│      test/cool_loss       │    0.3603402376174927     │
│    test/cool_prob_loss    │    0.7497891783714294     │
│      test/heat_loss       │    0.3301336169242859     │
│    test/heat_prob_loss    │    0.9582025408744812     │
│      test/total_loss      │    2.3984649181365967     │
└───────────────────────────┴───────────────────────────┘

## 2023.10.30

Finding the dropout could be more effective in validation test

- dropout = 0.5 while hidden_dim = 32

  | Name             | Type         | Params
--------------------------------------------------
0 | encoder          | LSTM         | 16.1 K
1 | heat_decoder     | Sequential   | 611   
2 | cool_decoder     | Sequential   | 611   
3 | heat_decoder_seq | LSTM         | 16.9 K
4 | cool_decoder_seq | LSTM         | 16.9 K
5 | quantile_loss    | QuantileLoss | 0     
6 | heat_tigger      | Sequential   | 577   
7 | cool_trigger     | Sequential   | 577   
--------------------------------------------------
52.3 K    Trainable params
0         Non-trainable params
52.3 K    Total params
0.209     Total estimated model params size (MB)


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        global_step        │         3641256.0         │
│      test/cool_loss       │    0.2821897566318512     │
│    test/cool_prob_loss    │    0.44244658946990967    │
│      test/heat_loss       │    0.2645571231842041     │
│    test/heat_prob_loss    │    0.35640254616737366    │
│      test/total_loss      │    1.3455960750579834     │
└───────────────────────────┴───────────────────────────┘


## 2024.04.04

### Prepare for the E+ dataset

Questions for Kingsley.

1. Is there a way to get the simulation data without you actually modeling it?

2. What is the modification of the script you make when you do mechanic load, partial load, and ideal load? Like if I would like to reproduce what you have been simulate, is there a script (.sh, .py) that I can refer to?

3. I know that is modeling hearing and cooling by electricity, is it okay or valid to ignore other output and just focus on these two variables?

4. 15 mins vs 1 hour?

5. how do you generate setpoints.csv?

6. 

#### Result 

rnn_v2_hidden128_dropout8e-1
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        global_step        │         9388031.0         │
│      test/cool_loss       │    0.04305875301361084    │
│    test/cool_prob_loss    │   0.008808477781713009    │
│      test/heat_loss       │    0.0668027251958847     │
│    test/heat_prob_loss    │    0.02083391137421131    │
│      test/total_loss      │    0.13950397074222565    │
└───────────────────────────┴───────────────────────────┘
