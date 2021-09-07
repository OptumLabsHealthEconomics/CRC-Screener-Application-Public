# globals.py

CANCER_TREATMENT_COSTS_11, CANCER_TREATMENT_COSTS_12, CANCER_TREATMENT_COSTS_13, CANCER_TREATMENT_COSTS_14 = 35565.94,  2863.32,  72877.60,  17810.16
CANCER_TREATMENT_COSTS_21, CANCER_TREATMENT_COSTS_22, CANCER_TREATMENT_COSTS_23, CANCER_TREATMENT_COSTS_24 = 50423.21,  3482.91,  82218.31,  19155.98
CANCER_TREATMENT_COSTS_31, CANCER_TREATMENT_COSTS_32, CANCER_TREATMENT_COSTS_33, CANCER_TREATMENT_COSTS_34 = 71769.60,  5607.20,  85067.77,  26649.66
CANCER_TREATMENT_COSTS_41, CANCER_TREATMENT_COSTS_42, CANCER_TREATMENT_COSTS_43, CANCER_TREATMENT_COSTS_44 = 104434.9,  28232.22, 105846.83, 65305.00

COST_DIAGNOSTIC_COLO, COST_SURVEILLANCE_COLO, COST_SCREENING_COLO, COST_SYMPTOM_COLO = 1337.46, 1337.46, 1337.46, 1337.46

# Utility inputs
ADJ_COLO = -0.0055 # per event
ADJ_COLO_COMPLICATION = -0.0384 # per event (any complication)
# stage1_init, stage_1_cont, stage_1-term_cancer, stge_1_term_other
TRMT_UTILITY_11, TRMT_UTILITY_12, TRMT_UTILITY_13, TRMT_UTILITY_14 = -0.15, -0.10, -0.29, -0.10
TRMT_UTILITY_21, TRMT_UTILITY_22, TRMT_UTILITY_23, TRMT_UTILITY_24 = -0.15, -0.10, -0.29, -0.10
TRMT_UTILITY_31, TRMT_UTILITY_32, TRMT_UTILITY_33, TRMT_UTILITY_34 = -0.15, -0.10, -0.29, -0.10
TRMT_UTILITY_41, TRMT_UTILITY_42, TRMT_UTILITY_43, TRMT_UTILITY_44 = -0.34, -0.29, -0.29, -0.29
# 18-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75+
# 0.924, 0.912, 0.889, 0.855, 0.830, 0.817, 0.755
AGE_UTILITY_18_24, AGE_UTILITY_25_34, AGE_UTILITY_35_44, AGE_UTILITY_45_54 = 0.924, 0.912, 0.889, 0.855
AGE_UTILITY_55_64, AGE_UTILITY_65_74, AGE_UTILITY_75 = 0.830, 0.817, 0.755


## SCREENER SETUP ##

valInvasive, valNonInvasive = 0, 1

INTERVAL_COLO, ADHERENCE_COLO = 10.0, 1.0
SENS_SMALL_COLO, SENS_MED_COLO, SENS_LARGE_COLO = 0.75, 0.85, 0.95
SCREEN_KIND_COLO = str('invasive')
TYPE_SCREEN_COLO = valInvasive
SENS_CANCER_COLO, SPEC_COLO = 0.95, 0.86
COST_COLO = 1337.46
SCREENING_AGE_MIN_COLO, SCREENING_AGE_MAX_COLO = 50, 80


INTERVAL_CG, ADHERENCE_CG = 3.0, 1.0
SENS_SMALL_CG, SENS_MED_CG, SENS_LARGE_CG = 0.172, 0.172, 0.424
SENS_CANCER_CG, SPEC_CG = 0.923, 0.898
COST_CG = 508.87
SCREENING_AGE_MIN_CG, SCREENING_AGE_MAX_CG = 50, 80
TYPE_SCREEN_CG = valNonInvasive
SCREEN_KIND_CG = str('stool')

INTERVAL_FIT, ADHERENCE_FIT = 1.0, 1.0
SENS_SMALL_FIT, SENS_MED_FIT, SENS_LARGE_FIT = 0.076, 0.076, 0.238
SENS_CANCER_FIT, SPEC_FIT = 0.738, 0.964
COST_FIT = 23.28
SCREENING_AGE_MIN_FIT, SCREENING_AGE_MAX_FIT = 50, 80
TYPE_SCREEN_FIT = valNonInvasive
SCREEN_KIND_FIT = str('stool')

INTERVAL_NT, ADHERENCE_NT = 0,0
SENS_SMALL_NT, SENS_MED_NT, SENS_LARGE_NT = 0,0,0
SENS_CANCER_NT, SPEC_NT = 0,0
COST_NT = 0
SCREENING_AGE_MIN_NT, SCREENING_AGE_MAX_NT = 0,0
TYPE_SCREEN_NT = valNonInvasive
SCREEN_KIND_NT = str('stool')

ADHERENCE_FOLLOW_UP_COLO = 1.0

## DISEASE PARAMETER SETUP ##

# Parameters for adenoma generation
POLYP_GEN_ALPHA_0 = -6.6
POLYP_GEN_SIGMA_ALPHA = 1.1
POLYP_GEN_ALPHA_1 = -0.24
POLYP_GEN_ALPHA_2K_45, POLYP_GEN_ALPHA_2K_65, POLYP_GEN_ALPHA_2K_75, POLYP_GEN_ALPHA_2K_120 = 0.037, 0.031, 0.029, 0.03
POLYP_GEN_SUM_AJDIFF_45, POLYP_GEN_SUM_AJDIFF_65, POLYP_GEN_SUM_AJDIFF_75, POLYP_GEN_SUM_AJDIFF_120 = 0.0, 0.3, 0.42, 0.35

# Parameters for adenoma growth
polyp_growth_cbeta1, polyp_growth_cbeta2, polyp_growth_rbeta1, polyp_growth_rbeta2 = 24.3, 1.8472066, 11.086583, 2.4485203

# Parameters for staging detected cancer
csp11, csp21, csp31 = 4.45118078, 0.29839421, 0.41562304
csp12, csp22, csp32 = -0.8845893, 0.16069904, 0.10670585
csp13, csp23, csp33 = 0.02665437, -0.015023,  -0.0169

## PEOPLE SETUP ##
# People are represented by an array. Attributes will be set during natural history. For each screening scenario, a copy of 'people' will
# be made and additional attributes will be filled in during screening/a realization of the simulaion
LIFE_TABLE_TYPE = 'cohort'#'period'
MAX_AGE = 120
BIRTH_YEAR = 1980

# People attributes
indGender = 0
genderMale, genderFemale = 0, 1
indAge = 1
indAlive = 2
indAgeOtherDeath = 3
indPolypRisk = 4
indNumActivePolyps = 5
indNumCancers = 6
indAgeNextSurvCol = 7
indAgeNextScreen = 8
indAgeDiagnosed = 9
indAgeCancerDeath = 10
indStageDiagnosed = 11
indLocDiagnosed = 12 
indReasonDiagnosed = 13 
diagSymptoms, diagIntervention = 1, 2
indNumSurveillanceColos = 14
indNumScreens = 15
indNumDiagColos = 16
indNumSymptColos = 17
indAgeComplicationDeath = 18
indCostScreen = 20
indCostColonoscopy = 21
indCostComplications = 22
indCostTreatment = 23
indLYAdjustment = 24

num_person_id = 25


## PEOPLE/POLYP SETUP ##
# polyps will be 3d array (a,b,c). [a] is number of people. [b] is number of possible polyps. So each person has a matrix of polyps which are all null
# until an adenoma forms.
global indSizePolyp         # Size of polyp. Only used in screening component
indSizePolyp = 0
global indLocationPolyp     # Location of polyp. 0 if polyp has not been created. 1-6 indicates location (see below for key)
indLocationPolyp = 1
global indAgeInitPolyp      # Age of person at which the polyp developed
indAgeInitPolyp = 2
global indGrowthRatePolyp   # Growth rate of this polyp
indGrowthRatePolyp = 3
global indSizeAtCancer      # Size at which this polyp will transition to preclinical crc
indSizeAtCancer = 4
global indIsCancer          # 0 if not yet cancer. If > 0, then marks the index+1 of the corresponding cancer in the cancer arrays.
indIsCancer = 5             # So if this is polyp polyps[person,i] has spawned a cancer, then ( cancers[person,polyps[person,i,indIsCancer]-1] ) is the corresponding cancer
global indIsActive          # Whether the polyp is active. 0 if adenomas is not null or has spawned a carcinoma. Else, 1. Only used in natural history
indIsActive = 6
global indAgeAtTransition   # Age of person at which the polyps transitions to preclinical crc
indAgeAtTransition = 7

num_polyp_id = 8

global numPolypsAllowed
numPolypsAllowed = 40

# Key for locations of adenomas/carcinomas
global locRectum, locSigmoid, locDescending, locTransverse, locAscending, locCecum
locRectum, locSigmoid, locDescending, locTransverse, locAscending, locCecum = 1, 2, 3, 4, 5, 6


## PEOPLE/CANCER SETUP ##
# cancers will be 3d array (a,b,c). [a] is number of people. [b] is number of possible cancers. So each person has a matrix of cancers which are all null
# until a carcinoma forms.

global indLocationCancer    # location of cancer
indLocationCancer = 0
global indAgeCancerInit     # age of person at which this cancer formed
indAgeCancerInit = 1
global indCancerGrowthRate  # growth rate of this cancer
indCancerGrowthRate = 2
global indSojournTCancer    # sojourn time of this cancer
indSojournTCancer = 3
global indSizeAtSymptoms    # cancer size at which symptoms will develop
indSizeAtSymptoms = 4
global indStageAtSymptoms   # cancer stage when symptoms develop
indStageAtSymptoms = 5
global indAgeAtSymptoms     # person age at which symptoms develop. Should be  == AgeCancerInit + SojournTCancer
indAgeAtSymptoms = 6
global indSurvivalYearsSymptoms # years of cancer-related survival if this cancer is detected from symptoms, relative to date of symptom onset
indSurvivalYearsSymptoms = 7
global indSizeAtDetect      # Size at which cancer is detected. If cancer is not detected with colonoscopy, 0, else, the sizee
indSizeAtDetect = 8
global indStageAtDetect     # If detected with colonoscopy, the cancer stage at that time
indStageAtDetect = 9
global indAgeAtDetect       # If detected with colonoscopy, the age of person when that occurs
indAgeAtDetect = 10
global indSurvivalYearsDetect # If detected with colonoscopy, years of cancer-related survival relative to date of diagnosis
indSurvivalYearsDetect = 11

num_cancer_id = 12

global numCancersAllowed
numCancersAllowed = 4
