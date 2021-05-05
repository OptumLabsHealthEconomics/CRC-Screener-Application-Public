# crcaim.py
import time
import numpy as np
import pandas as pd
from math import erf
from scipy.stats import norm
import numba
#import warnings
#warnings.filterwarnings('error')

# START HERE
def run(n):
    '''Setup of parameters, runs natural history, then overlays colonoscopy and cologuard screening.
    Simulates for n individuals

    Right now, everything is hardcoded and not generalized. Parameters are set as globals

    Returns dataframes of results summary for each screening scenario
    '''

    tr = time.time() #for timing

    ## DISEASE PARAMETER SETUP ##

    # Parameters for adenoma generation
    global polyp_gen_alpha_0, polyp_gen_sigma_alpha, polyp_gen_alpha_1, polyp_gen_alpha_2k, polyp_gen_sum_ajdiff
    polyp_gen_alpha_0 = -6.6
    polyp_gen_sigma_alpha = 1.1
    polyp_gen_alpha_1 = -0.24
    polyp_gen_alpha_2k = np.array([0.037, 0.031, 0.029, 0.03])
    polyp_gen_sum_ajdiff = np.array([0.0, 0.3, 0.42, 0.35])
    
    # Parameters for adenoma growth
    global polyp_growth_cbeta1, polyp_growth_cbeta2, polyp_growth_rbeta1, polyp_growth_rbeta2
    polyp_growth_cbeta1, polyp_growth_cbeta2, polyp_growth_rbeta1, polyp_growth_rbeta2 = 24.3, 1.8472066, 11.086583, 2.4485203

    # Parameters for staging detected cancer
    intercepts = np.array([4.45118078, 0.29839421, 0.41562304])
    root_size_terms = np.array([-0.8845893, 0.16069904, 0.10670585])
    size_terms = np.array([0.02665437, -0.015023,  -0.0169])
    global CANCER_STAGE_PARS
    CANCER_STAGE_PARS = np.array([intercepts, root_size_terms, size_terms]).T
    
    ## PEOPLE SETUP ##
    # People are represented by an array. Attributes will be set during natural history. For each screening scenario, a copy of 'people' will
    # be made and additional attributes will be filled in during screening/a realization of the simulaion

    global popSize
    popSize = n

    # Array for storing costs per person per year. For every screening scenario, a copy will be made
    costs = np.zeros((n,120))

    # People attributes
    global indGender            # gender. 0 or 1
    indGender = 0
    global genderMale, genderFemale
    genderMale, genderFemale = 0, 1
    global indAge               # Age. starts at 0
    indAge = 1
    global indAlive             # Whether tthey are alive. 1 if true, 0 once dead
    indAlive = 2
    global indAgeOtherDeath     # Age at which they will die of other causes. Determined in advance
    indAgeOtherDeath = 3
    global indPolypRisk         # Person-specific adenoma risk
    indPolypRisk = 4
    global indNumActivePolyps   # Number of active polyps (if there are n polyps, but one spawns a cancer, there are n-1 active polyps)
    indNumActivePolyps = 5
    global indNumCancers        # Number of cancers
    indNumCancers = 6
    global indAgeNextSurvCol    # Age at which they get their next surveillance colonoscopy. 0 if one is not scheduled
    indAgeNextSurvCol = 7
    global indAgeNextScreen     # Age a which they are scheduled to get their next screen. 0 if one is not scheduled
    indAgeNextScreen = 8
    global indAgeDiagnosed      # Age at which they received a cancer diagnosis. 0 if they haven't
    indAgeDiagnosed = 9
    global indAgeCancerDeath    # Age at which they will die of cancer. 0 if they haven't received cancer diagnosis. Can be nonzero even if they will die of other causes
    indAgeCancerDeath = 10
    global indStageDiagnosed    # Stage of cancer at diagnosis. 0 if they have not received diagnosis
    indStageDiagnosed = 11
    global indLocDiagnosed      # Location of diagnosed cancer. 0 if not diagnosed. 1-6 indicates location (see below for key)
    indLocDiagnosed = 12 
    global indReasonDiagnosed   # Reason for diagnosis. 0 if not diagnosed. 1 for symptoms, 2 for screening/surveillance detection
    indReasonDiagnosed = 13 
    global diagSymptoms, diagIntervention
    diagSymptoms, diagIntervention = 1, 2
    
    people = np.zeros((n,14))

    ## PEOPLE/POLYP SETUP ##
    # polyps will be 3d array (a,b,c). [a] is number of people. [b] is number of possible polyps. So each person has a matrix of polyps which are all null
    # until an adenoma forms.

    global indSizePolyp         # Size of polyp
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
    global indIsActive          # Whether the polyp is active. 0 if adenomas is not null or has spawned a carcinoma. Else, 1
    indIsActive = 6
    global indAgeAtTransition   # Age of person at which the polyps transitions to preclinical crc
    indAgeAtTransition = 7

    polyps = np.zeros((n,50,8)) # max 50 polyps. This can be easily changed though

    ## PEOPLE/POLYP SETUP ##
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

    cancers = np.zeros((n,4,12))  #max 3 cancers

    # Key for locations of adenomas/carcinomas
    global locRectum, locSigmoid, locDescending, locTransverse, locAscending, locCecum
    locRectum, locSigmoid, locDescending, locTransverse, locAscending, locCecum = 1, 2, 3, 4, 5, 6


    ## Filling in values ##

    people[:,indGender] = np.random.rand(popSize)
    # TODO: make a non-5050 split for gender. Get from census!
    males, females = people[:,indGender] > .5, people[:,indGender] <= .5
    people[males,indGender] = genderMale
    people[females,indGender] = genderFemale

    # Other-cause mortality
    # Life table for 1980 cohort, given as conditional probabilities of death
    life_table = pd.read_csv('1980_cohort.csv')
    life_table = life_table[['Age','M','F']].values
    life_table[:,1:] = 1-np.cumprod(1-life_table[:,1:], axis=0)
    life_table[-1,1:] = 1
    # determine years of other-cause death for each person
    people = get_other_death_dates(people, life_table)

    people[:,indAge] = 0.0
    people[:,indAlive] = 1.0
    # Determine individual polyp risk for each person
    people[:,indPolypRisk] = np.random.normal(loc=np.ones(popSize)*polyp_gen_alpha_0, scale=np.ones(popSize)*polyp_gen_sigma_alpha)

    # This was used to make the cdf of the generalized log distribution. Here for reference
    #cancer_size_cdf = np.array([[x/10,gen_log_pdf(x/10)] for x in range(140*10)])
    #cancer_size_cdf[:,1] = np.cumsum(cancer_size_cdf[:,1])
    #cancer_size_cdf[:,1] = cancer_size_cdf[:,1] / cancer_size_cdf[-1,1]
    


    ##############################
    ## SIMULATE NATURAL HISTORY
    ##############################
    people, polyps, cancers = natural_history(people, polyps, cancers)
    
    results = {}
    ### aggregate into df
    for i, person in enumerate(people):
        person_result = {}
        person_cancers = cancers[i]

        # determine cause of death
        person_cancers = person_cancers[person_cancers[:,indAgeAtSymptoms] > 0]
        person_cancers = person_cancers[person_cancers[:,indAgeAtSymptoms] < person[indAgeOtherDeath]]

        # only looking at cancers that were diagnosed prior to other-cause death
        death_ages = person_cancers[:,indAgeAtSymptoms] + person_cancers[:, indSurvivalYearsSymptoms]
        person_result['Other_Death_Age'] = person[indAgeOtherDeath]
        if len(person_cancers) > 0:
            # had some cancers
            arg_cancer = np.argmin(death_ages)
            final_cancer = person_cancers[arg_cancer]

            person_result['Cancer_Death_Age'] = min(death_ages)

            # active_cancers will be used to determine stage (might be more advanced tumor at diagnosis than the one that leads to first death)
            active_cancers = person_cancers[person_cancers[:,indAgeCancerInit] < min(person_cancers[:,indAgeAtSymptoms])] # just cancers that were active at/prior to first possible symptoms

            if death_ages[arg_cancer] < person[indAgeOtherDeath]:
                person_result['Death_Cause'] = 'cancer'
                person_result['Death_Age'] = death_ages[arg_cancer]
            else:
                person_result['Death_Cause'] = 'other'
                person_result['Death_Age'] = person[indAgeOtherDeath]
            # recording info about cancer if they were symptomatic prior to death (regardless of which cause of death).
            # One cancer is recorded here, and is the one which lead to symptoms prior to death and (would have) had the earliest cancer-caused of all cancers
            # (not necessarily first to become symptomatic)
            person_result['Stage'] = max(active_cancers[:,indStageAtSymptoms]) #final_cancer[indStageAtSymptoms]
            person_result['Size'] = final_cancer[indSizeAtSymptoms] # what to do here?
            person_result['Location'] = final_cancer[indLocationCancer]
            person_result['AgeSymptomatic'] = min(person_cancers[:,indAgeAtSymptoms])#final_cancer[indAgeAtSymptoms]
        else:
            person_result['Death_Cause'] = 'other'
            person_result['Death_Age'] = person[indAgeOtherDeath]
            
        
        results[i] = person_result

    results = pd.DataFrame.from_dict(results, orient='index')
    #print('cancer rate: ', np.mean(results.Stage > 0) ) 
    #print('cancer death rate: ', np.mean(results.Death_Cause == 'cancer'))
    #print('cancer death rate | cancer: ', np.mean( results[results.Stage > 0].Death_Cause == 'cancer' ) )
    #print('done natural history ', time.time()-tr)

    ##############################
    ## SIMULATE COLONOSCOPY SCREENING
    ##############################
    people_colo = people.copy()
    polyps_colo = polyps.copy()
    cancers_colo = cancers.copy()
    costs_colo = np.zeros((n,120))

    people_colo[:,indAge] = 0
    people_colo[:, indAlive] = 1
    people_colo, polyps_colo, cancers_colo, costs_colo = run_screening(people_colo, polyps_colo, cancers_colo, costs_colo, "colonoscopy")

    results_colo = {}
    ### aggregate into df
    for i, person in enumerate(people_colo):
        person_result = {}

        if person[indAgeCancerDeath] > 0:
            person_result['Cancer_Death_Age'] = person[indAgeCancerDeath]
        # determine cause of death
        if (person[indAgeOtherDeath] > person[indAgeCancerDeath]) & (person[indAgeCancerDeath] > 0):
            person_result['Death_Cause'] = 'cancer'
            person_result['Death_Age'] = person[indAgeCancerDeath]
            person_result['Diag_Reason'] = 'Symptoms' if person[indReasonDiagnosed] == diagSymptoms else 'Intervention'
        else:
            person_result['Death_Cause'] = 'other'
            person_result['Death_Age'] = person[indAgeOtherDeath]
        person_result['Diag_Age'] = person[indAgeDiagnosed]
        person_result['Diag_Stage'] = person[indStageDiagnosed]
        person_result['Diag_Location'] = person[indLocDiagnosed]           
            
        person_result['Costs'] = np.sum(costs_colo[i])
        
        results_colo[i] = person_result

    results_colo = pd.DataFrame.from_dict(results_colo, orient='index')

    ##############################
    ## SIMULATE COLOGUARD SCREENING
    ##############################
    people_cg = people.copy()
    polyps_cg = polyps.copy()
    cancers_cg = cancers.copy()
    costs_cg = np.zeros((n,120))

    people_cg[:,indAge] = 0
    people_cg[:, indAlive] = 1
    people_cg, polyps_cg, cancers_cg, costs_cg = run_screening(people_cg, polyps_cg, cancers_cg, costs_cg, "mtsDNA")

    results_cg = {}
    ### aggregate into df
    for i, person in enumerate(people_cg):
        person_result = {}

        if person[indAgeCancerDeath] > 0:
            person_result['Cancer_Death_Age'] = person[indAgeCancerDeath]
        # determine cause of death
        if (person[indAgeOtherDeath] > person[indAgeCancerDeath]) & (person[indAgeCancerDeath] > 0):
            person_result['Death_Cause'] = 'cancer'
            person_result['Death_Age'] = person[indAgeCancerDeath]
            person_result['Diag_Age'] = person[indAgeDiagnosed]
            person_result['Diag_Stage'] = person[indStageDiagnosed]
            person_result['Diag_Location'] = person[indLocDiagnosed]
            person_result['Diag_Reason'] = 'Symptoms' if person[indReasonDiagnosed] == diagSymptoms else 'Intervention'
        else:
            person_result['Death_Cause'] = 'other'
            person_result['Death_Age'] = person[indAgeOtherDeath]
        person_result['Costs'] = np.sum(costs_cg[i])
        
        results_cg[i] = person_result

    results_cg = pd.DataFrame.from_dict(results_cg, orient='index')

    return results, results_colo, results_cg


@numba.njit
def get_other_death_dates(people, dist):
    '''Determines and sets the ages at which each person will die from other causes, given a pre-sampled cdf

    Could vectorize this or find a slicker way of doing it for each person, but this isn't really a bottleneck so...
    '''
    for _, person in enumerate(people):
        x = np.random.random()
        idx = 1 if person[indGender] == genderMale else 2
        for val in dist:
            if val[idx] >= x:
                age = val[0]
                break
        person[indAgeOtherDeath] = age
    return people


# NOTE
def natural_history(people, polyps, cancers):
    ''' Runs and creates the natural history of each person, independent of any medical intervention of any kind. Multiple "parallel universes" are
    simulated, meaning even if a person would otherwise have gotten and died from cancer, we keep simulating progression of other adenomas/carcinomas
    This allows a direct comparison of results at the individual level for multiple different screeners without the need to resimulate the natural history
    '''

    idx_alive = people[:, indAlive] == 1
    people_age = 0
    while np.any(idx_alive) and people_age < 120:

        '''
        NOTE: the ordering of events here is important. I have it as
            - create new polyps
            - progress polyps/create new cancers
            - progress cancers
        This means it is possible for a polyp to be created and grow (even progress to cancer) within one year.
        In CRC-AIM, are events ordered in this way, or is the above edge case structurally not possible?
        '''
        
        # Each person that is not dead from other/noncancer causes has the chance to develop new polyps
        # Determine how many new polyps each person gets this year
        lambdas = get_poisson_lambda(people[idx_alive], people_age)
        num_new_polyps = np.zeros(popSize)
        num_new_polyps[idx_alive] = np.random.poisson(lambdas) # can't numba poisson (meaning we can't numba this whole routine :( -- Not worth it to write own method to sample from poisson from unif or exp variates)

        idx_gets_polyps = num_new_polyps >= 1 # indices of people who develop new polyps
        # For each person with new polyps, create new polyps, setup each new polyp
        people[idx_gets_polyps], polyps[idx_gets_polyps] = setup_polyps(
            people[idx_gets_polyps], polyps[idx_gets_polyps], num_new_polyps[idx_gets_polyps])
        
        # Each person that has polyps and is not dead will have a progression of their disease
        idx_has_polyps = (people[:,indNumActivePolyps] > 0.5) & idx_alive
        people[idx_has_polyps], polyps[idx_has_polyps], cancers[idx_has_polyps] = progress_polyps(people[idx_has_polyps], polyps[idx_has_polyps], cancers[idx_has_polyps], people_age)
        
        # A cancer is new if it has an age_init == people_age (it was initiated this year)
        idx_has_new_cancers = idx_alive & np.any(cancers[:,:,indAgeCancerInit] == people_age, axis=1) # people with at least one new developed cancer
        # For each person with new cancer, set each cancer up
        # It may just be worth doing the entire cancer setup in progress_polyps funcion (call it 'progress_disease' and do everything?)
        cancers[idx_has_new_cancers] = setup_cancers(people[idx_has_new_cancers], cancers[idx_has_new_cancers], CANCER_STAGE_PARS)#, cancer_size_cdf)

        # A person may die of other/noncancer causes
        newDeaths = (people[:, indAge] >= people[:, indAgeOtherDeath]) & (people[:, indAlive] == 1)
        people[newDeaths, indAlive] = 0

        # End of step

        # Determine who is still alive
        idx_alive = people[:, indAlive] == 1
        # Everyone still alive gets one year older
        people[idx_alive, indAge] += 1
        people_age += 1

    return people, polyps, cancers


@numba.njit
def get_poisson_lambda(people, people_age):
    '''Returns the lambda/rate parameter for poisson ademona generation
    '''
    
    if people_age < 45:
        age_bracket = 0
    elif people_age < 65:
        age_bracket = 1
    elif people_age < 75:
        age_bracket = 2
    else:
        age_bracket = 3

    risks = people[:,indPolypRisk] #ALPHA_0_i
    gender_risks = people[:,indGender] * polyp_gen_alpha_1 # gender*ALPHA_1
    age_risks = people[:,indAge] * polyp_gen_alpha_2k[age_bracket] # age*ALPHA_2K[age_bracket]
    extra_age_risks = polyp_gen_sum_ajdiff[age_bracket] # SUM_AJdiff[age_bracket]
    #lambda_i = ALPHA_0_i + gender*ALPHA_1 + age*ALPHA_2K[age_bracket] + SUM_AJdiff[age_bracket]
    lambdas = risks + gender_risks
    lambdas += age_risks
    lambdas += extra_age_risks
    lambdas = np.exp(lambdas)

    return lambdas


# NOTE
@numba.njit
def setup_polyps(people, polyps, num_new_polyps):
    '''Given people the number of new polyps a person should develop this time step, create instances of each polyp

    This requires that all input arrays are reduced to only the people who get at least one new adenoma this time step
    '''
    
    for i, person in enumerate(people):
        num_ever_polyps = np.sum(polyps[i,:,indAgeInitPolyp] > 0)
        if num_ever_polyps == polyps.shape[1]:
            # person already has max number of polyps. Don't create any new ones
            continue

        for p in range(int(num_new_polyps[i])):
            polyp_id = num_ever_polyps + p # if there were n polyps before, the next p should be at idx (n+p-1) [and p starts counting up from 0]
            if polyp_id > polyps.shape[1] - 1:
                # has hit max number of polyps already. Don't create any new ones
                continue
            
            # mark this polyp as active
            polyps[i, polyp_id, indIsActive] = 1
            person[indNumActivePolyps] += 1
            # age of initiation of this polyp is the persons current age
            polyps[i, polyp_id, indAgeInitPolyp] = person[indAge]
            
            # determine polyp location
            x = np.random.random()
            if x < .09:
                # P(rectum) -> 0.09
                location = locRectum # rectum
            elif x < .33:
                # P(sigmoid) -> 0.24
                location = locSigmoid # sigmoid
            elif x < .45:
                # P(descending) -> 0.12
                location = locDescending  # descending
            elif x < .69:
                # P(transverse) -> 0.24
                location = locTransverse  # transverse
            elif x < .92:
                # P(ascending) -> 0.23
                location = locAscending  # ascending
            else:
                # P(cecum) -> 0.08
                location = locCecum # cecum
            polyps[i, polyp_id, indLocationPolyp] = location

            # determine time for polyp to grow to 10mm
            if location == locRectum:
                # rectum
                b1 = polyp_growth_rbeta1
                b2 = polyp_growth_rbeta2
            else:
                # anywhere in colon
                b1 = polyp_growth_cbeta1
                b2 = polyp_growth_cbeta2
            
            t10mm = b1 * (-np.log(np.random.random())) ** (-1/b2) # time to 10mm

            # determine growth rate of this polyp
            dmax = 50.0
            dmin = 1.0
            growth_rate = -np.log((dmax - 10)/(dmax - dmin)) / t10mm
            polyps[i, polyp_id, indGrowthRatePolyp] = growth_rate

            '''
            NOTE: The cdf of size-at-cancer is quoted in CRC-AIM paper as 
                PHI( ( log(gamma_{1cm}*s) + gamma_{2cm}*(a-50) )/gamma_3 )
            This is equivalent to
                PHI( ( log(s) + log(gamma_{1cm}) + gamma_{2cm}*(a-50) )/gamma_3 )
            which is the cdf of lognormal with mean = ( -log(gamma_1) - gamma_2*(a-50) ), and stdev = (gamma_3).

            So we can generate the size at transition via a random lognormal with the quoted mean and stdev
            
            Is this correct?
            '''
            # determine size at preclinical-crc transition/carcinoma creation
            if person[indGender] == genderFemale: 
                # female
                if location == locRectum: 
                    #rectum
                    gamma1, gamma2 = .0470747, .0161731
                else: 
                    # colon
                    gamma1, gamma2 = .0444762, .0089362
            else:
                # male
                if location == locRectum:
                    #rectum
                    gamma1, gamma2 = .0472322, .0173598
                else: 
                    # colon
                    gamma1, gamma2 = .04, .0089232
            gamma3 = .5
            mean = -np.log(gamma1) - gamma2*(polyps[i,polyp_id,indAgeInitPolyp] - 50)
            
            # crc-aim uses a cycle-based approach, and thus uses probability of transition within each interval/time-step
            # conditioned on no transition in the past. It is more efficient to generate the size at cancer in advance
            polyps[i, polyp_id, indSizeAtCancer] = np.random.lognormal(mean, gamma3)             
    
    return people, polyps


@numba.njit
def progress_polyps(people, polyps, cancers, people_age):
    '''Ages/progresses all active polyps for all alive people with active polyps.
    Rather than storing the current size, we just determine if the new size is big enough such that it becomes cancer
    If it does become cancer, create a new cancer (but don't fully set it up), and link the polyp to that cancer, and mark the polyp as inactive (it is now cancer)
    '''
    
    for person_idx, person in enumerate(people):
        for polyp_idx, polyp in enumerate(polyps[person_idx]):            
            if polyp[indIsActive] == 0:
                # adenoma has become a carcinoma or has not been created, so go to next polyp
                continue 

            rate, t1 = polyp[indGrowthRatePolyp], (people_age - polyp[indAgeInitPolyp]) + 1 # age of polyp is people-age - age-at-polyp-init
            d_0, d_inf = 1.0, 50.0
            # Determine new polyp size by the end of this timestep (at t+1)
            d_t1 = d_inf - (d_inf - d_0) * np.exp(-rate * t1) 
            # If polyp grows to a size bigger than the pre-sampled size at preclinical CRC transition, polyp becomes a cancer
            if d_t1 >= polyp[indSizeAtCancer]:
                # Mark polyp as inactive
                polyp[indIsActive] = 0
                person[indNumActivePolyps] -= 1
                person[indNumCancers] += 1
                if person[indNumCancers] >= cancers.shape[1]: #reached max num of cancers
                    # we don't want to model a new cancer, but for now we will keep track of number of cancers beyond the max, to see if we need to adjust the maximum allowed
                    break

                # need to idx for new cancer.
                cancer_idx = int(person[indNumCancers] - 1) # we subtract because we have already incremented numcancers.
                polyps[person_idx, polyp_idx, indIsCancer] = cancer_idx + 1 # indicates this polyp has a cancer at indx polyp[indIsCancer]-1 in cancer array, and polyp doesn't need to be updated anymore
                polyps[person_idx, polyp_idx, indAgeAtTransition] = people_age # the age at transition is rounded to the integer age of tthe person at the beginning of this time step
                cancers[person_idx, cancer_idx, indLocationCancer] = polyp[indLocationPolyp] # cancer has same location as polyp
                cancers[person_idx, cancer_idx, indAgeCancerInit] = people_age


    return people, polyps, cancers


@numba.njit
def setup_cancers(people, cancers, cancer_stage_pars):
    '''Given people with new cancers and the cancers which have been "initialized" but not "set up", determine key variables for each cancer
    '''
    for idx_person, person in enumerate(cancers):
        for idx_cancer, cancer in enumerate(person):
            if (cancer[indSojournTCancer] > 0) | (cancer[indAgeCancerInit] == 0):
                # either (not a new cancer), or is an old cancer
                continue
            # Determine sojourn time
            if cancer[indLocationCancer] == locRectum:
                # rectum
                xi, nu = 1.148838, 0.564791
            else:
                # colon
                xi, nu = 0.943347, 0.492673
            sojourn_time = np.random.lognormal(xi, nu)
            cancer[indSojournTCancer] = sojourn_time
            # Determine size at symptoms/clinical diagnosis
            #size = get_cancer_size_from_ecdf(cancer_size_cdf)
            size = get_cancer_size_beta()
            cancer[indSizeAtSymptoms] = size
            # Determine cancer growth rate
            cancer[indCancerGrowthRate] = (2*size)**(1/sojourn_time) # 2*size is equivalent to size/.5 = size/min_size
            # Determine cancer stage at clinical detection
            cancer[indStageAtSymptoms] = get_cancer_stage_given_size(size, cancer_stage_pars)
            # Determine persons age at clinical diagnosis
            cancer[indAgeAtSymptoms] = people[idx_person, indAge] + sojourn_time
            # Determine time from diagnosis to death from from this specific cancer, should that happen
            cancer[indSurvivalYearsSymptoms] = get_death_from_cancer_age(people[idx_person,indAge], people[idx_person,indGender], cancer[indLocationCancer], cancer[indStageAtSymptoms])
            
    return cancers


# NOTE
@numba.njit
def get_cancer_size_beta():
    '''Generates a random size at which a single CRC is clinically detected

    NOTE: CRC-AIM dsecribes the pdf of a "generalized log distribution" which is used for this purpose.
    I believe there is a typo in the model description. The pdf is implemented in the function "gen_log_pdf".
    Upon analysis of this function, it doesn't seem to be a probability distribution. It does not integrate to 1
    within the bounds of (.5,140) or globally. Additionally, it is not a matter of finding a scaler to fix the integration issue,
    as the plot of this function with the given parameters is not quite the same as the curve displayed in fig s28.
    I have looked through some books and literature and can't find any distribution that is similar to the "generalized log distribution"
    used in CRC-AIM. Am I missing something? Is there a typo? Parameter issue?

    In the meantime, I am using a scaled ~beta(6*49/140, 6) scaled/shifted to interval [.5,140]. Parameters were found such that the mean and
    .25/.5/75 quantiles are similar to those quoted in fig s28. It's not perfect. Just a placeholder until the Glog dist is figured out
    '''
    
    beta = 6.0
    alpha = beta*49/140
    c_size = np.random.beta(alpha,beta) * 140
    if c_size <= .5: # minimum size is .5. If that is sampled, draw again
        c_size = get_cancer_size_beta()
    return c_size


# NOTE
def gen_log_pdf(s):
    '''Not used

    NOTE: This computes the pdf of the "generalized log distribution" for CRC size at clinical detection.

    See the NOTE in 'get_cancer_size_beta()'

    When plotted and integrated, it clearly does not describe a probability distribution with the same physical properties
    as the one in fig s28
    '''
    mu, sigma, lambd = 3.91048, 0.37775, 28.9135
    root = np.sqrt(s**2 + lambd**2)
    scaling = (s + root) / sigma
    scaling = scaling / (root**2 + s*root)
    normal_input = ( np.log((s + root)/2) - mu )/sigma
    #normal = (1 + erf(normal_input/np.sqrt(2)))/2
    normal = norm.cdf(normal_input)
    return scaling * normal


@numba.njit
def get_cancer_size_from_ecdf(cancer_size_cdf):
    '''Not used
    
    Given an array of the cdf sampled at a number of evenly spaced points, sample from the distribution
    '''
    x = np.random.random()
    size = cancer_size_cdf[0,0]
    for p, prob in enumerate(cancer_size_cdf):
        if prob[1] > x:
            size = cancer_size_cdf[p-1,0]
            break
        if prob[1] == 1:
            size = p
    return size


# NOTE
@numba.njit
def get_cancer_stage_given_size(c_size, cancer_stage_pars):
    ''' Randomly sample the stage of a cancer given the size in mm

    NOTE: I think these two formulas in crc-aim docs have typos?
    The formula on page 12 for the probabilities is wrong? It says probability of stage k is 
        exp(g_k)/(sum(g_m)_1^k).
    We aren't given parameters to compute g_4. So probability of stage K is actually 
        exp(g_k)/(1 + (sum(g_m)_1^k)), 
    which we compute for k=1,2,3, then infer for k=4

    Another thing is that on line 223/pg 16 of manuscript, and in the github, it says the logit function is 
        g_k = alpha_k + beta_k * size**-.5 + gamma_k * size
    However, in the supporing materials, table s34 suggests that it is actually 
        g_k = alpha_k + beta_k * size**.5 + gamma_k * size
    
    Notice the difference of 1/sqrt(size) vs sqrt(size)
    With some tests, it seems the latter is more likely to be right?
    '''

    S = np.array([1.0, np.sqrt(c_size), c_size])
    G = cancer_stage_pars.dot(S)
    pis = np.exp(G)
    cum_pis = 1 + np.sum(pis)
    
    prob = 0
    stage = 4
    x = np.random.random()
    for i, pi in enumerate(pis):
        prob += pi/cum_pis
        if x < prob:
            stage = i + 1
            break

    return stage


@numba.njit
def get_death_from_cancer_age(age, gender, location, stage):
    '''Determines the survival time of an individual after a cancer diagnosis, in years
    The survival models hard coded here are only from the 2000-2003 data. The assumption is we will never use
    the 1970s survival curves, because we are only modeling present/future outcomes/trends

    TODO: clean this function up with some vectorization for determination of distribution parameters/hard-code parameters outside this funtion

    If the cancer is stage 1 in colon:
        weibull F(t) = 1 - exp(-(t/lambda[scale])**k[shape])
    If the cancer is stage 1 in rectum:
        loglogistic F(t) = PHI((log(t) - mu)/sigma), PHI(x) = 1/(1+exp(-x))
    For all other stages and locations, a lognormal is used
    '''

    if stage == 1:
        if location == locRectum:
            # inverse cdf of loglogistic is ((1-r)/r)**sigma * exp(mu) where r is runiform
            # stage 1 rectal
            x = np.random.random()
            sigma = 1.1079
            intercept = 4.2475
            age_effect = 0.4703 if age < 50 else .7033 if age < 60 else .1593 if age < 70 else -0.2561 if age < 80 else -1.0768
            time_from_diagnosis = ((1-x)/x)**sigma * np.exp(intercept + age_effect)
        else:
            # took inverse of given cdf, because here is always confusion on different forms of weibull. Numpy implements different one than crc-aim I think
            # stage 1 colon
            x = np.random.random()
            shape = 0.699099
            intercept = 5.8797
            age_effect = 1.6097 if age < 50 else 0.6499 if age < 60 else -0.0115 if age < 70 else -0.4863 if age < 80 else -1.7619
            sex_effect = 0.1321 * (1 if gender == genderFemale else -1)
            scale = intercept + age_effect + sex_effect
            time_from_diagnosis = scale * (-np.log(1-x))**(shape)            
    else:
        if stage == 2:
            if location == locRectum:
                sigma = 2.1874
                intercept = 3.4680
                age_effect = 0.9608 if age < 50 else 0.4976 if age < 60 else 0.1146 if age < 70 else -0.3914 if age < 80 else -1.1817
                sex_effect = 0
            else:
                sigma = 2.8316
                intercept = 4.5013
                age_effect = 0.6193 if age < 50 else 0.3778 if age < 60 else 0.3243 if age < 70 else -0.2226 if age < 80 else -1.0987
                sex_effect = 0.1825 * (1 if gender == genderFemale else -1)
        elif stage == 3:
            if location == locRectum:
                sigma = 1.7519
                intercept = 2.3558
                age_effect = 0.4700 if age < 50 else 0.3899 if age < 60 else 0.3039 if age < 70 else -0.1072 if age < 80 else -1.0567
                sex_effect = 0
            else:
                sigma = 2.2040
                intercept = 2.5361
                age_effect = 0.5416 if age < 50 else 0.4263 if age < 60 else 0.2067 if age < 70 else -0.1463 if age < 80 else -1.0283
                sex_effect = 0
        else:
            if location == locRectum:
                sigma = 1.4349
                intercept = -0.1707
                age_effect = 0.5883 if age < 50 else 0.5408 if age < 60 else 0.0489 if age < 70 else -0.3437 if age < 80 else -0.8343
                sex_effect = 0
            else:
                sigma = 1.5583
                intercept = -0.3766
                age_effect = 0.6531 if age < 50 else 0.3190 if age < 60 else 0.0861 if age < 70 else -0.2930 if age < 80 else -0.7653
                sex_effect = 0
        mu = intercept + age_effect + sex_effect
        time_from_diagnosis = np.random.lognormal(mu, sigma)

    return time_from_diagnosis


@numba.njit
def run_screening(people, polyps, cancers, costs, screen_type):
    '''Overlays screening with a given screen type onto the presimulated natural history
    All information about natural history is encapsulated within tthe people/polyps/cancer arrays. These need to be copies of the originals,
    as they will be overwritten.

    Array of costs is also input, which is (num_people, 120) array, which will contain the costs per person per age/year
    '''

    idx_alive = people[:, indAlive] == 1
    people_age = 0
    while np.any(idx_alive) and people_age < costs.shape[1]: #if we change shape of costs array, this will need to change
        if people_age < 20:
            # we are restricting cancer diagnoses of any kind to ages 20+
            newOtherDeaths = (
                idx_alive & 
                (people[:, indAge] == people[:, indAgeOtherDeath]) # BUG opportunity: if ageotherdeath is non-integer, this needs to be changed
                )
            people[newOtherDeaths, indAlive] = 0
            idx_alive = (people[:,indAlive] == 1)
            people[idx_alive, indAge] += 1
            people_age += 1
            continue
        if people_age == 45:
            # First screening occurs.
            # This can be generalized/user settable in future. For now, just assume screening occurs at age 45
            people[idx_alive, indAgeNextScreen] = people_age

        ## Surveillance colonoscopies, if due
        idx_surveillance = (
            idx_alive & 
            (people[:, indAgeNextSurvCol] == people_age)
            )
        people[idx_surveillance], polyps[idx_surveillance], cancers[idx_surveillance], costs[idx_surveillance] = get_surveillance_colos(people[idx_surveillance], polyps[idx_surveillance], cancers[idx_surveillance], costs[idx_surveillance])
        
        ## Screening occurs, if due
        idx_screening = (
            idx_alive & 
            (people[:,indAgeNextScreen] == people_age)
            )
        # the structure will need to change here in the future to be more flexible. For now, there are two options: standard colonoscopy and Cologuard
        if screen_type == 'mtsDNA':
            people[idx_screening], polyps[idx_screening], cancers[idx_screening], costs[idx_screening] = get_screened_stool(people[idx_screening], polyps[idx_screening], cancers[idx_screening], costs[idx_screening])
        elif screen_type == 'colonoscopy':
            people[idx_screening], polyps[idx_screening], cancers[idx_screening], costs[idx_screening] = get_screened_colo(people[idx_screening], polyps[idx_screening], cancers[idx_screening], costs[idx_screening])
        
        ## People may become symptomatic/receive clinical diagnosis
        idx_has_symptomatic_cancers = (people[:,indAge] == -4000) # Not really a more efficient way to create boolean array of this size with numba
        # This is not vectorized because the axis argument in np.any() is not supported with numba which means that 
        # when a matrix is provided, it reduces to a scalar and not a vector/array, so a loop approach must be taken instead :(
        for i, pcancers in enumerate(cancers):
            idx_has_symptomatic_cancers[i] = (
                (people[i,indAgeDiagnosed] == 0) & # agediagnosed is set right after this
                np.any( 
                    (pcancers[:,indAgeCancerInit] > 0) & 
                    (pcancers[:,indAgeCancerInit] <= people_age) & 
                    (pcancers[:,indAgeAtSymptoms] <= people_age) 
                    )
            )
        idx_has_symptomatic_cancers = idx_has_symptomatic_cancers & idx_alive
        people[idx_has_symptomatic_cancers, indAgeDiagnosed] = people_age
        people[idx_has_symptomatic_cancers], cancers[idx_has_symptomatic_cancers] = diagnose_cancer_symptomatic(people[idx_has_symptomatic_cancers], cancers[idx_has_symptomatic_cancers])

        ## Diagnosis by screen
        idx_diagnosed_screen = (
            (people[:,indAgeDiagnosed] == people_age) & 
            (~idx_has_symptomatic_cancers)
            )
        people[idx_diagnosed_screen], cancers[idx_diagnosed_screen] = diagnose_cancer_screen(people[idx_diagnosed_screen], cancers[idx_diagnosed_screen], CANCER_STAGE_PARS)

        ## Add treatment costs
        # TODO: for each cancer patient, add appropriate cancer costs

        ## Other cause death
        newOtherDeaths = (
            idx_alive & 
            (people[:, indAge] >= people[:, indAgeOtherDeath])
            ) # should this be ageotherdeath + 1? bc it takes non-intteger values?
        people[newOtherDeaths, indAlive] = 0
        ## Cancer death
        newCancerDeaths = (
            idx_alive & 
            ((people[:,indAgeDiagnosed] > 0) & (people[:, indAge] >= people[:, indAgeCancerDeath]))
            )
        people[newCancerDeaths, indAlive] = 0
        ## End of step
        ## Get one year older
        idx_alive = (people[:, indAlive] == 1)
        people[idx_alive, indAge] += 1
        people_age += 1
    
    return people, polyps, cancers, costs


@numba.njit
def get_surveillance_colos(people, polyps, cancers, costs):
    '''Surveillance Colonoscopies
    '''
    for i, person in enumerate(people):
        costs[i, int(person[indAge])] += 1300 #(THIS IS JUST A TEMP MARKER)
        people[i], polyps[i], cancers[i] = get_colonoscopy(person, polyps[i], cancers[i])
    return people, polyps, cancers, costs


# NOTE
@numba.njit
def get_colonoscopy(person, polyps, cancers):
    ''' A person gets a colonoscopy (of any type)
    Also right now, no complications occur, and full reach is assumed

    NOTE: it is noted that if a colonoscopy does not have full reach, a second is performed. Is this recursion indefinite? Could someone get 5
        colonoscopies in a row because each does not have full reach? What is the cost sructure if this happens? Are they all the same cost?
        Does this apply for all types of colonoscopies (screen, surveillance, diagnostic)? Is there perfect adherence to these subsequent colonoscopies?
    
    NOTE: The supplementary material in Knudsen says that complications can only occur if a polypectomy is performed. I vaguely remember in one of our workshops
        that complications may arise even if no polypectomy is performed. Is this true? I also remember a comment about death being a possible outcome
        of a colonoscopy complication, but it isn't perfectly clear how that is determined in Knudsen (and citations) and in CRC-AIM. How is probabilitty of death computed?
    
    NOTE: If a person is found to have cancer but no polyps, do we assume a polypectomy was performed?
    '''

    SENSITIVITY_COLONOSCOPY_CANCER, SENSITIVITY_COLONOSCOPY_SMALL, SENSITIVITY_COLONOSCOPY_MED, SENSITIVITY_COLONOSCOPY_LARGE = .95, .75, .85, .95
    SPECIFICITY_COLONOSCOPY = .86

    next_surv_colo = 10 # 10 indicates they will have normal screening in 10 years
    gets_polypectomy = False
    # BUG for right now, assume colonoscopy has FULL reach. No repeats if only partial reach
    
    idx_active_cancers = (cancers[:,indAgeCancerInit] <= person[indAge]) & (cancers[:, indAgeCancerInit] > 0)
    idx_active_polyps = (polyps[:,indAgeInitPolyp] <= person[indAge]) & (polyps[:,indAgeInitPolyp] > 0) & (polyps[:, indIsCancer] == 0)

    if np.any(idx_active_cancers):
        # has cancer
        num_active = int(np.sum(idx_active_cancers))
        detected = (np.random.random(num_active) < SENSITIVITY_COLONOSCOPY_CANCER)
        if np.any(detected):
            # full diagnosis routines will be performed later in time step
            person[indAgeDiagnosed] = person[indAge]
            gets_polypectomy = True # does this happen?
            next_surv_colo = 0 # will indicate that are diagnosed. No surveillance or screening

    if np.any(idx_active_polyps) and person[indAgeDiagnosed] == 0: #active polyps and weren't just diagnosed with cancer
        rands = np.random.random(int(np.sum(idx_active_polyps)))
        detected = np.zeros(len(rands))
        idx_small = polyps[idx_active_polyps, indSizePolyp] < 6
        idx_med = (~idx_small) & (polyps[idx_active_polyps, indSizePolyp] < 10)
        idx_large = (polyps[idx_active_polyps, indSizePolyp] >= 10)
        detected[idx_small] = (rands[idx_small] < SENSITIVITY_COLONOSCOPY_SMALL)
        detected[idx_med] = (rands[idx_med] < SENSITIVITY_COLONOSCOPY_MED)
        detected[idx_large] = (rands[idx_large] < SENSITIVITY_COLONOSCOPY_LARGE)
        ## ASSIGN SURVEILLANCE SCHEDULE BASED ON SIZES OF THESE ARRAYS!
        idx_detected = (detected == True) # this is redundant...
        # for each polyp detected, resect it, meaning indicate it is no longer cancer. Should we just zero it out? Will that break anything?
        # Then also need to delete any cancers that would have come from this polyp
        if np.any(idx_detected):
            gets_polypectomy = True
        
        # potential BUG: make sure correct cancers are being deleted
        for ind in polyps[idx_detected, indIsCancer]:
            cancers[int(ind) - 1] = 0
        polyps[idx_detected] = 0 # DELETES POLYPS THAT ARE DETECTED and RESECTED
        if (np.sum(detected) >= 3) | (np.sum(detected[idx_large]) >= 1): # at least one > 10mm or at least 3 of any size
            # next colo in 3 years
            next_surv_colo = 3
        elif np.any(idx_detected):
            # next colo in 5 years
            next_surv_colo = 5
        else:
            # return to normal screening test, next screen in 10 YEARS
            next_surv_colo = 10
            pass

    if (~ np.any(idx_active_cancers)) & (~np.any(idx_active_polyps)):
        # no active disease
        if np.random.random() > SPECIFICITY_COLONOSCOPY:
            # FP
            gets_polypectomy = True
            
    if next_surv_colo == 0:
        # diagnosed with cancer. No more screening or surveillance
        person[indAgeNextScreen] = 0
        person[indAgeNextSurvCol] = 0
        person[indAgeDiagnosed] = person[indAge]
    elif next_surv_colo == 10:
        # Screening resumes in 10 years
        person[indAgeNextScreen] = person[indAge] + 10
        person[indAgeNextSurvCol] = 0
    else:
        # Surveillance is scheduled
        person[indAgeNextSurvCol] = person[indAge] + next_surv_colo
        person[indAgeNextScreen] = 0
    
    return person, polyps, cancers


@numba.njit
def get_screened_colo(people, polyps, cancers, costs):
    '''Screening with colonoscopy
    '''
    for i, person in enumerate(people):
        year = int(person[indAge])
        costs[i, year] += 1300 #cost of regular colonoscopy? (THIS IS JUST A TEMP MARKER)
        person[:], polyps[i,:,:], cancers[i,:,:] = get_colonoscopy(person, polyps[i], cancers[i])
    return people, polyps, cancers, costs


@numba.njit
def get_screened_stool(people, polyps, cancers, costs):
    '''People get screened via cologuard. We are assuming perfect adherence and followup compliance
    Assume screen sensitivity is based on most advanced lesion and results in colonoscopy if positive -- fine assumption here, but if generalized beyond cologuard, might need to change
    '''
    # COLOGUARD
    TEST_SENSITIVITY_CANCER = .923
    TEST_SENSITIVITY_LARGE = .424
    TEST_SENSITIVITY_MED = .172
    TEST_SENSITIVITY_SMALL = .172
    TEST_SPECIFICITY = .898
    SCREEN_INTERVAL = 3
    SCREEN_COST = 300 # place holder
    DIAG_COLO_COST = 3000 # place holder

    for i, person in enumerate(people):
        age = int(person[indAge])
        followup = False
        # determine most advanced lesion
        if np.any(cancers[i,:,indAgeCancerInit] <= age):
            if np.random.random() < TEST_SENSITIVITY_CANCER:
                # TP. Get followup colonoscopy
                followup = True
            else:
                # FN. Assign time to next screen
                pass
        elif np.any(polyps[i,:,indAgeInitPolyp] <= age):
            max_size = np.max(polyps[(polyps[i,:,indAgeInitPolyp] <= age), indSizePolyp])
            sensitivity = TEST_SENSITIVITY_LARGE if max_size >= 10 else TEST_SENSITIVITY_MED if max_size >= 6 else TEST_SENSITIVITY_SMALL
            if np.random.random() < sensitivity:
                # TP. Get followup colonoscopy
                followup = True
            else:
                # FN. Assign time to next screen
                pass
        else:
            # no disease
            if np.random.random() > TEST_SPECIFICITY:
                # FP. Get followup colonoscopy
                followup = True
            else:
                # TN. Assign time to next screen
                pass
        
        if followup:
            # get diagnostic colonscopy
            costs[i,age] += DIAG_COLO_COST
            person[:], polyps[i,:], cancers[i,:] = get_colonoscopy(person,polyps[i],cancers[i])
        else:
            person[indAgeNextScreen] = age + SCREEN_INTERVAL # nothing was found. Screening resumes as normal

    # add cost of screen
    costs[:, age] += SCREEN_COST
    return people, polyps, cancers, costs


# NOTE
@numba.njit
def diagnose_cancer_symptomatic(people, cancers):
    '''Diagnosis of people who have symptomatic cancers
    Stage and size and death time have already been computed/determined, but we need to make sure that other cancers 
    are "killed" and such (maybe I should kill the polyps too? -- don't think it matters for the way things run now)

    NOTE: there is ambiguity about which cancer determines survival if multiple are existence at diagnosis. Lets say a person at age 60 is 
        given a diagnosis (from screen) of cancer. They have two carcinomas at that time: cancer (A) that would have become 
        symptomatic at age 61 and another (B) at age 62. There are several edges cases that aren't clear to me. 
        - Is the stage of diagnosis the stage of the cancer with greatest stage at age 60?
        - Which cancer determines survival? If (A) and (B) are stage 2 and 3 respectively (at screening detection) is (B) chosen at the diagnosed cancer, and new survival time sampled?
        - What if (A) is sampled to have shorter survival time than (B)? Does that matter?
        - The supplementary material says that "survival functions will not be implemented until [the sojourn time expires]". I am not sure what this means
            - Lets say a person is diagnosed with stage 2 cancer. The cancer would have other become symptomatic in 3.1 years (stage 4).
                The new survival time for the stage 2 cancer is sampled as 1.1 years. Is the new time att cancer death 3.1 + 1.1 = 4.2 years? As in we wait 
                for sojourn time to expire, and then start counting down the sampled survival years?
                Or is the new time 3.1 years, because the person should die before that, but we don't allow the person to die until after sojourn time expires?
                So two cases: [cancer death age] = [sympt age] + [surv years], or [cancer death age] = max({[sympt age], [age] + [survyears]})          
    '''

    for i, person in enumerate(people):
        person_cancers = cancers[i]
        # determine which cancers are currently in existence
        idx_active = (person_cancers[:,indAgeCancerInit] <= int(person[indAge])) & (person_cancers[:,indAgeCancerInit] > 0)
        # diagosed cancer stage goes to the cancer with highest stage
        person[indStageDiagnosed] = np.max(cancers[i][idx_active,indStageAtSymptoms])
        # if multiple cancers are in existense, we need to determine which
        death_ages = person_cancers[idx_active,indAgeCancerInit] + person_cancers[idx_active,indSojournTCancer] + person_cancers[idx_active,indSurvivalYearsSymptoms]
        death_age = np.min(death_ages)
        symptom_ages = person_cancers[idx_active,indSojournTCancer] + person_cancers[idx_active,indAgeCancerInit]
        symptom_age = np.min(symptom_ages)
        person[indAgeCancerDeath] = max((death_age, symptom_age)) # if cancer death is before end of dwell/sojourn time, wait until symptomatic to die from cancer
        person[indStageDiagnosed] = np.max(person_cancers[idx_active,indStageAtSymptoms]) # stage is the max of active cancers, not the one that kills first
        person[indReasonDiagnosed] = diagSymptoms
    return people, cancers


@numba.njit
def diagnose_cancer_screen(people, cancers, cancer_stage_pars):
    '''Diagnosis procedures, following diagnoses from a colonoscopy
    '''

    for i, person in enumerate(people):
        # we don't keep track of which cancer was diagnosed/caught, as we assume all active cancers will be found. 
        # So get size/stage/survival for all active cancers
        idx_active = (cancers[i][:,indAgeCancerInit] <= person[indAge]) & (cancers[i][:,indAgeCancerInit] > 0)
        for j, cancer in enumerate(cancers[i]):
            if ((cancer[indAgeCancerInit] > person[indAge]) | (cancer[indAgeCancerInit] == 0)):
                continue
            # Compute size at diagnosis
            b = cancer[indCancerGrowthRate]
            a = .5 # min cancer size
            t = person[indAge] - cancer[indAgeCancerInit] # NOTE: this will always be an integer number of years. Is that okay??
            c_size =  a * b**t
            # Determine stage
            stage = get_cancer_stage_given_size(c_size, cancer_stage_pars)
            # Survival time form this cancer
            survival_years = get_death_from_cancer_age(person[indAge], person[indGender], person[indLocationCancer], stage)
            cancers[i,j,indSizeAtDetect] = c_size
            cancers[i,j,indStageAtDetect] = stage
            cancers[i,j,indSurvivalYearsDetect] = survival_years
        
        # Possible death ages
        death_ages = person[indAge] + cancers[i][idx_active,indSurvivalYearsDetect]
        death_age = np.min(death_ages)
        # Possible symptom ages
        symptom_ages = cancers[i][idx_active,indSojournTCancer] + cancers[i][idx_active,indAgeCancerInit]
        symptom_age = np.min(symptom_ages)

        # Cancer death age given possible death and symptom ages
        person[indAgeCancerDeath] = max((death_age, symptom_age)) # if cancer death is before end of dwell/sojourn time, wait until symptomatic to die from cancer
        person[indStageDiagnosed] = np.max(cancers[i][idx_active,indStageAtDetect]) # stage is the max of active cancers, not the one that kills first
        person[indLocDiagnosed] = cancers[i][np.argmax(cancers[i][idx_active,indStageAtDetect]),indLocDiagnosed]
        person[indReasonDiagnosed] = diagIntervention

    return people, cancers
