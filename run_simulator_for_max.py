import opt_comb_struct_space as ocss
import sys
import os
import datetime

# https://stackoverflow.com/questions/1432924/python-change-the-scripts-working-directory-to-the-scripts-own-directory
# Trick to make the file directory as the working directory
# This is used in aeolus where the script file is in a diferent folder
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# SIM_BASE_FOLDER = "../../../simulator/NoCsim_coreMapped_16thMay2017/"
SIM_BASE_FOLDER = "../simulator/NoCsim_coreMapped_16thMay2017_6"

NUM_PROCESS = 8
# sim_list =['dedup', 'canneal', 'lu']#, 'fft','radix', 'water', 'vips', 'fluid']
# sim_list = ['canneal']
# MAX_VAL_FOLDER_RLS_ORD = "../../../simulator/dummy_sim_9thJuly/max_val_rls_ord/"
# MAX_VAL_FOLDER_RLS_UNORD = "../../../simulator/dummy_sim_9thJuly/max_val_rls_unord/"

# MAX_VAL_FOLDER_STAGE_ORD = "../../../simulator/dummy_sim_9thJuly/max_val_stage_ord/"
# MAX_VAL_FOLDER_STAGE_UNORD = "../../../simulator/dummy_sim_9thJuly/max_val_stage_unord/"


config_list = [
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_12September_actual_sim_rf_rls_ord_0",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_12September_actual_sim_rf_rls_ord_1",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_12September_actual_sim_rf_rls_ord_2",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_12September_actual_sim_rf_rls_ord_4",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_12September_actual_sim_rf_rls_ord_5",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_12September_actual_sim_rf_stg_ord_6",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_12September_actual_sim_rf_stg_ord_8",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_12September_actual_sim_rf_stg_ord_9",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_12September_actual_sim_rf_stg_ord_10",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_12September_actual_sim_rf_stg_ord_11",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_13September_actual_sim_rf_rls_unord_0",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_13September_actual_sim_rf_rls_unord_1",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_13September_actual_sim_rf_rls_unord_2",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_13September_actual_sim_rf_rls_unord_4",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_13September_actual_sim_rf_rls_unord_5",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_13September_actual_sim_rf_stg_unord_6",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_13September_actual_sim_rf_stg_unord_8",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_13September_actual_sim_rf_stg_unord_9",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_13September_actual_sim_rf_stg_unord_10",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
{
	"MAX_VAL_FOLDER":"../simulator/12thSept/max_val_13September_actual_sim_rf_stg_unord_11",
	"SIM_LIST":['dedup', 'canneal', 'lu'],
},
]

for config in config_list:
	ocss.run_all_simulator_for_max(SIM_BASE_FOLDER, 
		config["MAX_VAL_FOLDER"], 
		config["SIM_LIST"], 
		NUM_PROCESS)

	
# ocss.run_all_simulator_for_max(SIM_BASE_FOLDER, MAX_VAL_FOLDER_RLS_ORD, sim_list, NUM_PROCESS)
# ocss.run_all_simulator_for_max(SIM_BASE_FOLDER, MAX_VAL_FOLDER_RLS_UNORD, sim_list, NUM_PROCESS)
# ocss.run_all_simulator_for_max(SIM_BASE_FOLDER, MAX_VAL_FOLDER_STAGE_ORD, sim_list, NUM_PROCESS)

# Variation with number of iterations 50
# [ 5.90845336  5.90845336  6.01110458  6.40239312  6.40239312  6.40239312
#   6.40239312  6.84113749  6.84113749  7.35898724  7.35898724  7.53187377
#   7.59564454  7.64697999  7.95998952  8.52264633  8.53530013  8.90795096
#   8.90795096  8.92674366  8.92674366  9.46661336  9.47583145  9.47583145
#   9.57442886  9.65528005  9.65528005  9.65528005  9.65528005  9.92682224
#  10.41656098 10.51800019 10.51800019 10.51800019 10.52647765 10.52647765
#  10.52647765 10.52647765 10.52647765 10.52647765 10.63835194 10.64116977
#  10.66012943 10.75425426 10.75425426 10.75425426 10.754505   10.754505
#  10.754505   10.754505   10.754505   10.89986203]
# Variation with number of iterations 150
# [ 2.64536708  4.40533645  5.16940109  5.20326692  6.03923921  6.98757942
#   6.98757942  7.07241749  7.07241749  7.07241749  7.1482159   7.28511783
#   7.51851724  8.28243984  8.49250472  8.49250472  8.66032468  8.68280308
#   8.68280308  8.98155842  9.04506598  9.04506598  9.04506598  9.21612897
#   9.64962512  9.64962512  9.64962512  9.64962512  9.64962512  9.64962512
#   9.7510056   9.7510056   9.84649966  9.84649966  9.97706866  9.97706866
#   9.97706866 10.11664031 10.13691198 10.13691198 10.56680236 10.56680236
#  10.58857985 10.76069896 10.80693053 10.85778731 10.85778731 11.03348241
#  11.03348241 11.22222574 11.22222574 11.22222574]
# Iteration 1000
# [ 3.16154431  3.22659964  3.59803254  4.10742456  4.58387147  4.86740965
#   5.50693533  5.83322328  5.83322328  6.21073175  6.21073175  6.21073175
#   6.43112169  6.748046    6.84170115  7.42117781  8.07037327  8.07037327
#   8.09695376  8.09695376  8.19729657  8.25478613  8.25478613  8.57081174
#   8.57081174  8.57081174  8.81840776  8.81840776  8.81840776  9.56388826
#   9.73772807  9.88291641  9.88291641  9.97745809  9.97745809  9.98010099
#   9.98127379  9.98127379  9.98127379  9.98127379 10.03288933 10.04835501
#  10.14731848 10.25918409 10.28028486 10.33277064 10.33277064 10.33314899
#  10.42339869 10.42339869 10.42339869 10.42883457]
# Iteration 500
# [ 1.91877798  3.79117177  4.49177928  4.49177928  4.6570414   4.6570414
#   6.20876911  6.47133941  6.47133941  6.82699323  6.90036104  6.90036104
#   7.07519685  7.07519685  7.18858758  7.92595192  8.22598158  8.55629969
#   8.55629969  8.55629969  8.55629969  8.74254322  8.74254322  8.74254322
#   9.08048059  9.08048059  9.26089016  9.26598156  9.26598156  9.3486731
#   9.41757415  9.56094489  9.56094489  9.91381993  9.91381993 10.10396239
#  10.10396239 10.20346977 10.20346977 10.57310439 10.66265292 10.68636224
#  10.68636224 10.72840574 10.72840574 10.72840574 10.72840574 10.72840574
#  10.86137685 10.86137685 10.88824575 10.88824575]
# Iteration 250
# [ 2.19586742  2.87765503  4.84243387  6.22645635  6.30594211  6.80554948
#   7.31945206  7.31945206  7.4405771   8.23709325  8.39904561  8.39904561
#   8.54975041  8.54975041  8.54975041  8.54975041  8.81482648  8.81482648
#   9.06578481  9.06578481  9.06578481  9.42229461  9.52105101  9.64644279
#   9.64644279  9.64644279  9.64644279  9.7691496   9.7691496   9.7691496
#   9.7691496   9.7691496   9.7691496   9.77462273  9.77462273  9.77462273
#   9.77699783  9.83313289  9.83313289  9.83313289  9.83313289  9.83313289
#   9.86771511  9.95953768 10.18515985 10.28441082 10.37403203 10.37403203
#  10.5967877  10.5967877  10.77922376 10.84304475]
# Iteration 200
# [ 2.17931373  2.48349627  2.93189785  4.52831903  4.52831903  4.79989289
#   4.87575984  5.9450012   6.09209347  6.7390321   6.96584147  6.96584147
#   7.4309921   7.4309921   7.4309921   7.4325913   7.95564226  8.1640325
#   8.35971484  8.60274572  8.63746805  8.71792674  8.99105479  9.21508688
#   9.21508688  9.52810875  9.58798629  9.60429748  9.7077586   9.7077586
#   9.7077586   9.7077586   9.84832832  9.84899894  9.84899894 10.11517244
#  10.12481569 10.12481569 10.29661127 10.29661127 10.31956305 10.31956305
#  10.35703875 10.39908224 10.4411314  10.4411314  10.4411314  10.4411314
#  10.4411314  10.51787792 10.51787792 10.51787792]
# Iteration 125
# [ 3.50612793  3.66901662  4.65213923  4.75617055  4.75617055  4.85287584
#   5.30483888  5.40606685  6.18503992  7.30256993  7.55663297  7.55663297
#   7.74156676  7.84554312  8.6663589   8.67043702  8.98474711  9.01637349
#   9.11760688  9.12561343  9.12561343  9.12561343  9.21105166  9.21105166
#   9.21105166  9.44337725  9.44337725  9.50692812  9.55080649  9.60768289
#   9.60768289  9.60768289  9.78024887  9.98145432 10.13400959 10.23544526
#  10.25721708 10.483277   10.50416849 10.76342741 10.76342741 10.76342741
#  10.76342741 10.76342741 10.76342741 10.79400421 10.89357462 10.89357462
#  10.89357462 10.91191446 10.91384629 10.91384629]

# Variation with K. Increasing K makes the average number of nodes expanded to increase
# Probably get the average number of iterations too
# K = 2
# [ 3.41903446  4.03456469  4.73599594  4.73599594  4.73599594  5.94089613
#   6.15207305  6.15207305  6.15207305  6.29755662  6.81879564  7.16750754
#   7.16750754  7.18849571  7.70914553  7.70914553  8.17159636  8.17159636
#   8.17159636  8.17159636  8.55672317  8.55672317  8.55672317  8.55672317
#   8.55672317  8.67778944  8.94641417  9.02837514  9.02837514  9.39881736
#   9.39881736  9.39881736  9.62095041  9.62095041  9.69744336  9.69744336
#   9.81243192  9.81243192  9.89861391  9.90554724  9.90554724  9.91202242
#  10.13937293 10.13937293 10.13937293 10.61285086 10.80199238 10.80199238
#  10.87605228 10.92330645 10.92330645 10.92330645]
# K = 0
# [ 3.07511456  5.53331696  5.53331696  5.69862142  5.73800718  6.78685895
#   6.78685895  7.46426292  7.46426292  7.70543623  7.70543623  7.70543623
#   7.70543623  8.58393473  8.83481468  8.83481468  8.83481468  9.19605818
#   9.19605818  9.659712    9.659712    9.659712    9.6686521   9.6686521
#  10.00101181 10.21274339 10.44936989 10.44936989 10.44936989 10.44936989
#  10.77072347 10.77072347 10.85516583 10.85516583 10.85516583 10.85516583
#  10.85516583 10.85516583 10.85516583 10.87211124 10.87211124 10.87211124
#  10.97746254 10.97746254 10.97746254 10.97746254 10.97746254 10.97746254
#  10.97746254 10.97746254 10.97746254 11.04471554]
# K=0.5
# [ 3.14233036  3.54889723  3.86445697  5.04602158  5.04602158  5.44678673
#   5.98499317  5.98499317  7.13094423  7.13094423  7.56588087  7.56588087
#   7.56588087  7.622304    8.15108532  9.23884678  9.23884678  9.24245502
#   9.24245502  9.45026174  9.45026174  9.62120857  9.98464642  9.98464642
#   9.98464642  9.98464642 10.01733447 10.01733447 10.01733447 10.10323483
#  10.10323483 10.10323483 10.10323483 10.12109408 10.29353543 10.29353543
#  10.60747105 10.60747105 10.60747105 10.65268208 10.71307108 10.71307108
#  10.71307108 10.71307108 10.71307108 10.71307108 10.71307108 10.79753089
#  10.79753089 10.79753089 10.79753089 10.79753089]
# K = 1.5
# [ 1.92213376  5.06968023  5.37751956  5.37751956  5.85554617  6.52377771
#   7.42781111  7.42781111  7.81306833  8.11233893  8.11233893  8.2225274
#   8.42167598  8.43814509  8.67777722  8.67777722  8.88384949  9.20674673
#   9.47510403  9.50043715  9.50043715  9.51913204 10.20098094 10.24823516
#  10.2842304  10.2842304  10.2842304  10.2842304  10.2842304  10.2842304
#  10.53758296 10.53758296 10.53758296 10.72369712 10.72369712 10.95211508
#  10.95211508 10.95211508 10.95211508 10.95211508 10.95211508 11.04794681
#  11.04794681 11.04794681 11.04794681 11.04794681 11.04794681 11.04794681
#  11.10389611 11.10898865 11.10898865 11.11092047]



