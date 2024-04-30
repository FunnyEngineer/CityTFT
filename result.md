# Training Result

## UT Campus

### RNN MSE (W/o prob)
#### Hidden Dim 32

  | Name              | Type       | Params
-------------------------------------------------
0 | encoder           | LSTM       | 16.1 K
1 | heat_lstm_decoder | LSTM       | 16.9 K
2 | cool_lstm_decoder | LSTM       | 16.9 K
3 | heat_decoder      | Sequential | 577   
4 | cool_decoder      | Sequential | 577   
-------------------------------------------------
51.1 K    Trainable params
0         Non-trainable params
51.1 K    Total params
0.204     Total estimated model params size (MB)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test/cool_diff       │       62248214528.0       │
│      test/cool_loss       │     6.029795169830322     │
│      test/heat_diff       │       11291580416.0       │
│      test/heat_loss       │    0.3344557285308838     │
│      test/total_diff      │       73539829760.0       │
│      test/total_loss      │    6.3642425537109375     │
└───────────────────────────┴───────────────────────────┘
#### Hidden Dim 64

  | Name              | Type       | Params
-------------------------------------------------
0 | encoder           | LSTM       | 56.8 K
1 | heat_lstm_decoder | LSTM       | 66.6 K
2 | cool_lstm_decoder | LSTM       | 66.6 K
3 | heat_decoder      | Sequential | 2.2 K 
4 | cool_decoder      | Sequential | 2.2 K 
-------------------------------------------------
194 K     Trainable params
0         Non-trainable params
194 K     Total params
0.777     Total estimated model params size (MB)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test/cool_diff       │       8479185408.0        │
│      test/cool_loss       │    0.12551981210708618    │
│      test/heat_diff       │       4933745152.0        │
│      test/heat_loss       │    0.08469194918870926    │
│      test/total_diff      │       13412921344.0       │
│      test/total_loss      │     0.210211843252182     │
└───────────────────────────┴───────────────────────────┘




### RNN Qunatile Loss
#### Hidden Dim 32

  | Name              | Type         | Params
---------------------------------------------------
0 | encoder           | LSTM         | 16.1 K
1 | heat_lstm_decoder | LSTM         | 16.9 K
2 | cool_lstm_decoder | LSTM         | 16.9 K
3 | heat_decoder      | Sequential   | 611   
4 | cool_decoder      | Sequential   | 611   
5 | quantile_loss     | QuantileLoss | 0     
6 | sigmoid           | Sigmoid      | 0     
---------------------------------------------------
51.1 K    Trainable params
0         Non-trainable params
51.1 K    Total params
0.205     Total estimated model params size (MB)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test/cool_diff       │       75762065408.0       │
│      test/cool_loss       │    1.8979088068008423     │
│      test/heat_diff       │       15166856192.0       │
│      test/heat_loss       │    0.3406160771846771     │
│      test/total_diff      │       90928832512.0       │
│      test/total_loss      │     2.238524913787842     │
└───────────────────────────┴───────────────────────────┘
#### Hidden Dim 64
  | Name              | Type         | Params
---------------------------------------------------
0 | encoder           | LSTM         | 56.8 K
1 | heat_lstm_decoder | LSTM         | 66.6 K
2 | cool_lstm_decoder | LSTM         | 66.6 K
3 | heat_decoder      | Sequential   | 2.2 K 
4 | cool_decoder      | Sequential   | 2.2 K 
5 | quantile_loss     | QuantileLoss | 0     
6 | sigmoid           | Sigmoid      | 0     
---------------------------------------------------
194 K     Trainable params
0         Non-trainable params
194 K     Total params
0.778     Total estimated model params size (MB)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test/cool_diff       │       55984709632.0       │
│      test/cool_loss       │    0.6495267748832703     │
│      test/heat_diff       │       10469953536.0       │
│      test/heat_loss       │    0.22154715657234192    │
│      test/total_diff      │       66454659072.0       │
│      test/total_loss      │    0.8710752725601196     │
└───────────────────────────┴───────────────────────────┘


### RNN Prob
#### Hidden Dim 32

  | Name              | Type         | Params
---------------------------------------------------
0 | encoder           | LSTM         | 16.1 K
1 | heat_lstm_decoder | LSTM         | 16.9 K
2 | cool_lstm_decoder | LSTM         | 16.9 K
3 | heat_decoder      | Sequential   | 611   
4 | cool_decoder      | Sequential   | 611   
5 | quantile_loss     | QuantileLoss | 0     
6 | sigmoid           | Sigmoid      | 0     
7 | heat_tigger       | Sequential   | 577   
8 | cool_trigger      | Sequential   | 577   
---------------------------------------------------
52.3 K    Trainable params
0         Non-trainable params
52.3 K    Total params
0.209     Total estimated model params size (MB)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        global_step        │         3641256.0         │
│      test/cool_diff       │       74991427584.0       │
│      test/cool_loss       │     1.332492709159851     │
│    test/cool_prob_loss    │    4.6100077629089355     │
│      test/heat_diff       │       21044596736.0       │
│      test/heat_loss       │    0.5679283142089844     │
│    test/heat_prob_loss    │    0.9492749571800232     │
│      test/total_diff      │       96035905536.0       │
│      test/total_loss      │     7.459692478179932     │
└───────────────────────────┴───────────────────────────┘

#### Hidden Dim 64

  | Name              | Type         | Params
---------------------------------------------------
0 | encoder           | LSTM         | 56.8 K
1 | heat_lstm_decoder | LSTM         | 66.6 K
2 | cool_lstm_decoder | LSTM         | 66.6 K
3 | heat_decoder      | Sequential   | 2.2 K 
4 | cool_decoder      | Sequential   | 2.2 K 
5 | quantile_loss     | QuantileLoss | 0     
6 | sigmoid           | Sigmoid      | 0     
7 | heat_tigger       | Sequential   | 2.2 K 
8 | cool_trigger      | Sequential   | 2.2 K 
---------------------------------------------------
198 K     Trainable params
0         Non-trainable params
198 K     Total params
0.795     Total estimated model params size (MB)


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        global_step        │         3641256.0         │
│      test/cool_diff       │       13438556160.0       │
│      test/cool_loss       │    0.4229707419872284     │
│    test/cool_prob_loss    │     1.148207426071167     │
│      test/heat_diff       │       4885055488.0        │
│      test/heat_loss       │    0.2898247241973877     │
│    test/heat_prob_loss    │    1.0507392883300781     │
│      test/total_diff      │       18323605504.0       │
│      test/total_loss      │    2.9117417335510254     │
└───────────────────────────┴───────────────────────────┘

### Transformer MSE

#### Hidden Dim 64
  | Name              | Type               | Params
---------------------------------------------------------
0 | encoder           | TransformerEncoder | 50.7 K
1 | heat_lstm_decoder | LSTM               | 66.6 K
2 | cool_lstm_decoder | LSTM               | 66.6 K
3 | heat_decoder      | Sequential         | 2.2 K 
4 | cool_decoder      | Sequential         | 2.2 K 
5 | linear_embed      | Linear             | 1.7 K 
6 | heat_decoder_seq  | TransformerEncoder | 562 K 
7 | cool_decoder_seq  | TransformerEncoder | 562 K 
---------------------------------------------------------
1.3 M     Trainable params
0         Non-trainable params
1.3 M     Total params
5.258     Total estimated model params size (MB)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test/cool_diff       │       7524467200.0        │
│      test/cool_loss       │    0.11138669401407242    │
│      test/heat_diff       │       4026339072.0        │
│      test/heat_loss       │    0.06911543011665344    │
│      test/total_diff      │       11550784512.0       │
│      test/total_loss      │    0.18050235509872437    │
└───────────────────────────┴───────────────────────────┘

### Transformer Quantile

  | Name              | Type               | Params
---------------------------------------------------------
0 | encoder           | TransformerEncoder | 50.7 K
1 | heat_lstm_decoder | LSTM               | 66.6 K
2 | cool_lstm_decoder | LSTM               | 66.6 K
3 | heat_decoder      | Sequential         | 2.2 K 
4 | cool_decoder      | Sequential         | 2.2 K 
5 | linear_embed      | Linear             | 1.7 K 
6 | heat_decoder_seq  | TransformerEncoder | 562 K 
7 | cool_decoder_seq  | TransformerEncoder | 562 K 
8 | quantile_loss     | QuantileLoss       | 0     
9 | sigmoid           | Sigmoid            | 0     
---------------------------------------------------------
1.3 M     Trainable params
0         Non-trainable params
1.3 M     Total params
5.259     Total estimated model params size (MB)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test/cool_diff       │       7735001600.0        │
│      test/cool_loss       │     0.211231991648674     │
│      test/heat_diff       │       4763931136.0        │
│      test/heat_loss       │    0.1749287098646164     │
│      test/total_diff      │       12498940928.0       │
│      test/total_loss      │    0.38616088032722473    │
└───────────────────────────┴───────────────────────────┘


### Transformer Prob

   | Name              | Type               | Params
----------------------------------------------------------
0  | encoder           | TransformerEncoder | 50.7 K
1  | heat_lstm_decoder | LSTM               | 66.6 K
2  | cool_lstm_decoder | LSTM               | 66.6 K
3  | heat_decoder      | Sequential         | 2.2 K 
4  | cool_decoder      | Sequential         | 2.2 K 
5  | linear_embed      | Linear             | 1.7 K 
6  | heat_decoder_seq  | TransformerEncoder | 562 K 
7  | cool_decoder_seq  | TransformerEncoder | 562 K 
8  | quantile_loss     | QuantileLoss       | 0     
9  | sigmoid           | Sigmoid            | 0     
10 | heat_tigger       | Sequential         | 2.2 K 
11 | cool_trigger      | Sequential         | 2.2 K 
----------------------------------------------------------
1.3 M     Trainable params
0         Non-trainable params
1.3 M     Total params
5.275     Total estimated model params size (MB)


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        global_step        │         3641256.0         │
│      test/cool_diff       │       8662501376.0        │
│      test/cool_loss       │    0.22884081304073334    │
│    test/cool_prob_loss    │    0.9991825819015503     │
│      test/heat_diff       │       4690141696.0        │
│      test/heat_loss       │    0.15103378891944885    │
│    test/heat_prob_loss    │    1.0152422189712524     │
│      test/total_diff      │       13352637440.0       │
│      test/total_loss      │     2.394296646118164     │
└───────────────────────────┴───────────────────────────┘

### TFT

#### Hidden 64

  | Name           | Type                   | Params
----------------------------------------------------------
0 | embedding      | LazyEmbedding          | 0     
1 | static_encoder | StaticCovariateEncoder | 359 K 
2 | TFTpart2       | RecursiveScriptModule  | 679 K 
3 | quantile_loss  | QuantileLoss           | 0     
----------------------------------------------------------
1.0 M     Trainable params
0         Non-trainable params
1.0 M     Total params
4.154     Total estimated model params size (MB)



#### Hidden 128

  | Name           | Type                   | Params
----------------------------------------------------------
0 | embedding      | LazyEmbedding          | 0     
1 | static_encoder | StaticCovariateEncoder | 1.4 M 
2 | TFTpart2       | RecursiveScriptModule  | 2.6 M 
3 | quantile_loss  | QuantileLoss           | 0     
----------------------------------------------------------
4.0 M     Trainable params
0         Non-trainable params
4.0 M     Total params
16.188    Total estimated model params size (MB)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        global_step        │         3641256.0         │
│      test/cool_diff       │       8334778368.0        │
│      test/cool_loss       │    0.3993500769138336     │
│    test/cool_prob_loss    │     1.528727650642395     │
│      test/heat_diff       │       4434494464.0        │
│      test/heat_loss       │    0.3427436947822571     │
│    test/heat_prob_loss    │    1.1852524280548096     │
│      test/total_diff      │       12769277952.0       │
│      test/total_loss      │    3.4560749530792236     │
└───────────────────────────┴───────────────────────────┘


### UnTFT

  | Name             | Type               | Params
--------------------------------------------------------
0 | static_encoder   | LinearBasicBlock   | 1.2 K 
1 | temporal_encoder | LinearBasicBlock   | 1.2 K 
2 | encoder          | LSTM               | 264 K 
3 | static_enricher  | LinearBasicBlock   | 37.8 K
4 | heat_decoder_seq | TransformerEncoder | 1.9 M 
5 | cool_decoder_seq | TransformerEncoder | 1.9 M 
6 | heat_decoder     | Sequential         | 12.5 K
7 | cool_decoder     | Sequential         | 12.5 K
8 | heat_tigger      | Sequential         | 12.5 K
9 | cool_trigger     | Sequential         | 12.5 K
--------------------------------------------------------
4.1 M     Trainable params
0         Non-trainable params
4.1 M     Total params
16.424    Total estimated model params size (MB)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        global_step        │         3641256.0         │
│      test/cool_diff       │       13356789760.0       │
│      test/cool_loss       │    0.3427762985229492     │
│    test/cool_prob_loss    │     1.974974274635315     │
│      test/heat_diff       │       6801024512.0        │
│      test/heat_loss       │    0.14468233287334442    │
│    test/heat_prob_loss    │    1.7384814023971558     │
│      test/total_diff      │       20157818880.0       │
│      test/total_loss      │     4.200916767120361     │
└───────────────────────────┴───────────────────────────┘
