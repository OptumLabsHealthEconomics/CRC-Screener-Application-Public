# crcaim.py
import time
import os
cwd = os.getcwd()
import numpy as np
import pandas as pd
from scipy.stats import norm
import numba
import pandas_gbq
import multiprocess as mp
import globs


def run_app(client, screener, window, n_jobs=1, limit=None):
    '''Overlays new technology screening over previously run natural history and compares outcomes to standard of care screeners
    Outcomes are computed for those alive and free of diagnosed cancer at age 40, EXCEPT for costs over time, which includes the whole population
    Most outcomes are per 1000. See code
    Parameters:
        client -- The [database].Client with permissions to read tables
        screener -- {Dict} The parameters of the screener that will be run against the standard-of-care screeners
        window -- {array-like} The ages at which screening starts and ends
        n_jobs -- {int} Number of parallel processes to spawn
        limit -- {int} If not None, the number of people to run (for debugging purposes). Else, runs 1 million
    
    Returns:
        results -- {Dict} The desired aggregate/population outcomes for all SOC and New Screener pathways
    '''

    new_tech = {
        'interval':screener['interval'],
        'adherence': screener['adherence'],
        'sensitivity_small': screener['sensitivity']['Polyps<6mm'],
        'sensitivity_med': screener['sensitivity']['6mm<Polyps<10mm'],
        'sensitivity_large': screener['sensitivity']['10mm<Polyps'],
        'sensitivity_cancer':screener['sensitivity']['CRC'],
        'specificity':screener['specificity'],
        'cost':screener['cost'],
        'kind':screener['kind'], # can be 'invasive' or 'noninvasive'
        'screening_age_min':window[0],
        'screening_age_max':window[1]
        }
    
    extra_pars = {
        'life_table':'cohort'#'period'
    }
    
    # write new tech parameters into globs module prior to compilation
    setup_globs(new_tech, extra_pars=extra_pars)

    t0 = time.time()
    # Read in the "people" array from pre-run natural history simulation
    people = get_nh_people(client, limit=limit)
    t0 = time.time()
    # Read in the "polyps" array from pre-run natural history simulation
    polyps = get_nh_polyps(len(people), client, limit=limit)
    t0 = time.time()
    # Read in the "cancers" array from pre-run natural history simulation
    cancers = get_nh_cancers(len(people), client, limit=limit)
    costs = np.zeros(globs.MAX_AGE)
    t0 = time.time()
    # Overlay the new screening protocol on simulated natural history
    people, polyps, cancers, costs = run_screening(people, polyps, cancers, costs, 'new technology', n_jobs=n_jobs)
    print('Done simulating: {:.2f} minutes'.format((time.time()-t0)/60))
    # Record results from screening simulation
    results_nt, costs_nt = make_results_df(people, costs, 'new technology')
    t0 = time.time()
    # Read in results from previously simulated SOC screeners
    results_none, costs_none, results_colo, costs_colo, results_cg, costs_cg, results_fit, costs_fit = get_SOC_stored_results(client)
    print('Done pulling SOC results: {:.2f} minutes'.format((time.time()-t0)/60))
    
    ##############
    # Aggregate and compare outcomes of SOC and New Technology
    # Results are written into dictionary
    ##############

    results = {}
    screeners = ['none', 'colonoscopy', 'cologuard', 'FIT', 'new_tech']

    all_res = [results_none, results_colo, results_cg, results_fit, results_nt]
    all_costs = [costs_none, costs_colo, costs_cg, costs_fit, costs_nt]

    def cost_per_capita(df, cost, age):
        if cost == 0:
            return 0
        else:
            num_alive = np.sum(df.Death_Age >= age)
            return cost / num_alive

    # COSTS over time (FOR ALL PEOPLE, including those dead or with cancer before age 40)
    temp = {}
    for costs, res, name in zip(all_costs, all_res, screeners):
        costs['Cost_per_Capita'] = costs.apply(lambda row: cost_per_capita(res, row.cost, row.year), axis=1)
        temp[name] = costs
    results['CostsTime'] = temp

    # Filter results down to just those alive and free of diagnosed cancer at age 40
    for i in range(len(all_res)):
        df = all_res[i]
        all_res[i] = df.loc[((df.Death_Age > 40) & ~(df.Diag_Age <= 40))]
    
    # num screens per 1000
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.NumScreens)*1000
    results['num_screens'] = temp

    # diagnosed cancers per 1000
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.Diag_Age > 0) * 1000
    results['crc_cases'] = temp

    # diagnosed cancers by stage per 1000
    for stage in range(1,5):
        temp = {}
        for name, df in zip(screeners, all_res):
            temp[name] = np.mean(df.Diag_Stage == stage) * 1000
        results['diag_stage_{}'.format(stage)] = temp

    # surveillance colos per 1000
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.NumSurvColos) * 1000
    results['surv_colos'] = temp

    # diagnostic colos per 1000
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.NumDiagColos) * 1000
    results['diag_colos'] = temp

    # symptom colos per 1000
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.NumSymptColos) * 1000
    results['sympt_colos'] = temp

    # total colonoscopies per 1000
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.NumColos) * 1000
    results['total_colos'] = temp
    
    # CRC deaths per 1000
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.Death_Cause == 'cancer') * 1000
    results['crc_deaths'] = temp

    # complication deaths per 1000
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.Death_Cause == 'complication') * 1000
    results['complication_deaths'] = temp

    # Deaths averted per 1000
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = (np.mean(df.Death_Cause == 'other') - np.mean(results_none.Death_Cause == 'other')) * 1000
    results['deaths_averted'] = temp

    # LYG per 1000
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = (np.mean(df.Death_Age) - np.mean(results_none.Death_Age)) * 1000
    results['LYG'] = temp

    # QALY
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.QALY)
    results['QALY'] = temp

    # QALYG per 1000
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = (np.mean(df.QALY) - np.mean(results_none.QALY)) * 1000
    results['QALYG'] = temp
    
    # Costs per member
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.Costs)
    results['perMemberCost'] = temp

    # ICER per 1000
    temp = {}
    for name in screeners:
        if name == 'none':
            temp[name] = 0
            continue
        temp[name] =  (results['perMemberCost'][name] - results['perMemberCost']['none']) / results['QALYG'][name]
    results['ICER'] = temp

    return results


def bq_query_table(sql, client):
    '''Uses given client to execute given query into a data frame
    Parameters:
        sql -- {str} The full query to execute (in BigQuery format)
        client -- {[database].Client}
    
    Returns:
        {pd.DataFrame} The query result
    ''' 
    result = client.query(sql).to_dataframe()
    return result


def get_SOC_stored_results(client):
    '''Query all stored simulated SOC screening results
    Returns per-person results and per year costs results for no screening, SOC colonoscopy, SOC cologuard, and SOC FIT screening
    '''

    results_none = bq_query_table("SELECT * FROM `[server].rdt_crc_screening_outcomes.SOC_results` WHERE Screener = 'none' ORDER BY person_id", client)
    results_colo = bq_query_table("SELECT * FROM `[server].rdt_crc_screening_outcomes.SOC_results` WHERE Screener = 'colonoscopy' ORDER BY person_id ASC", client)
    results_cg = bq_query_table("SELECT * FROM `[server].rdt_crc_screening_outcomes.SOC_results` WHERE Screener = 'cologuard' ORDER BY person_id ASC", client)
    results_fit = bq_query_table("SELECT * FROM `[server].rdt_crc_screening_outcomes.SOC_results` WHERE Screener = 'fit' ORDER BY person_id ASC", client)
    
    costs_none = bq_query_table("SELECT * FROM `[server].rdt_crc_screening_outcomes.SOC_results_costs` WHERE Screener = 'none' ORDER BY year ASC ", client)
    costs_colo = bq_query_table("SELECT * FROM `[server].rdt_crc_screening_outcomes.SOC_results_costs` WHERE Screener = 'colonoscopy' ORDER BY year ASC ", client)
    costs_cg = bq_query_table("SELECT * FROM `[server].rdt_crc_screening_outcomes.SOC_results_costs` WHERE Screener = 'cologuard' ORDER BY year ASC ", client)
    costs_fit = bq_query_table("SELECT * FROM `[server].rdt_crc_screening_outcomes.SOC_results_costs` WHERE Screener = 'fit' ORDER BY year ASC ", client)
    

    return results_none, costs_none, results_colo, costs_colo, results_cg, costs_cg, results_fit, costs_fit


def get_nh_people(client, limit=None):
    '''Queries for prerun natural history and converts data into required numpy array for screening simulation
    client -- The bigquery client
    limit -- Number of individuals to limit to, if debugging with smaller sample than what is stored in GCP
    Returns the (n,num_person_id) people array
    '''

    t0 = time.time()
    people_bq = bq_query_table('SELECT * FROM `[server].rdt_crc_screening_outcomes.nh_people`', client)
    print("Done pulling people: {:2f}".format((time.time()-t0)/60))
    t0 = time.time()
    if limit is not None:
        people_bq = people_bq.loc[people_bq.pid < limit]
    people = np.zeros((len(people_bq),globs.num_person_id))
    t0 = time.time()
    # should sort DF by pid (or do so in query, but need to make sure it retains order in resulting DF)
    # Then 
    #   people[:,globs.indAgeOtherDeath] = df.AgeOtherDeath.values
    # should do the trick. Do same with the others
    for i, row in people_bq.iterrows():
        pid = int(row.pid)
        people[pid,globs.indGender] = globs.genderMale if row.Gender == 'Male' else globs.genderFemale
        people[pid,globs.indAgeOtherDeath] = row.AgeOtherDeath
        people[pid,globs.indPolypRisk] = row.PolypRisk
        people[pid,globs.indAlive] = 1
    print("Done setting up people: {:2f}".format((time.time()-t0)/60))
    return people


def get_nh_polyps(num_people, client, limit=None):
    '''Queries for prerun natural history and converts data into required numpy array for screening simulation
    num_people -- {int} Number of presimulated people
    client -- The bigquery client
    limit -- Number of individuals to limit to, if debugging with smaller sample than what is stored in GCP
    Returns the (num_people,numPolypsAllowed,globs.num_polyp_id) polyps array
    '''

    t0 = time.time()
    polyps_bq = bq_query_table('SELECT * FROM `[server].rdt_crc_screening_outcomes.nh_polyps`', client)
    print("Done pulling polyps: {:2f}".format((time.time()-t0)/60))
    print("Done pulling polyps: {:2f}".format((time.time()-t0)/60))
    t0 = time.time()
    if limit is not None:
        polyps_bq = polyps_bq.loc[polyps_bq.person_id < limit]
    polyps = np.zeros((num_people, globs.numPolypsAllowed, globs.num_polyp_id))
    # possible speedup solution. sort by polyp_id, person_id. For each polyp_id, get list of person_ids = people_ids. Take values = vals (of a column at indX). 
    # Then can say polyps[people_ids, polyp_id, indX] = vals
    # Intuition tells me this will be tremendously faster. Loop is reduce to numPolypsAllowed.
    # Alternatively, could just takes values of polyps_bq, then pass to jitted function that uses a loop.
    for i, row in polyps_bq.iterrows():
        person_id = int(row.person_id)
        polyp_id = int(row.polyp_id)
        polyps[person_id,polyp_id,globs.indLocationPolyp] = row.Location
        polyps[person_id,polyp_id,globs.indAgeInitPolyp] = row.AgeInit
        polyps[person_id,polyp_id,globs.indGrowthRatePolyp] = row.GrowthRate
        polyps[person_id,polyp_id,globs.indSizeAtCancer] = row.SizeAtCancer
        polyps[person_id,polyp_id,globs.indIsCancer] = row.cancer_id
        polyps[person_id,polyp_id,globs.indAgeAtTransition] = row.AgeAtTransition
    print("Done setting up polyps: {:2f}".format((time.time()-t0)/60))
    return polyps


def get_nh_cancers(num_people, client, limit=None):
    '''Queries for prerun natural history and converts data into required numpy array for screening simulation
    num_people -- {int} Number of presimulated people
    client -- The bigquery client
    limit -- Number of individuals to limit to, if debugging with smaller sample than what is stored in GCP
    Returns the (num_people,numCancersAllowed,globs.num_cancer_id) cancers array
    '''

    t0 = time.time()
    cancers_bq = bq_query_table('SELECT * FROM `[server].rdt_crc_screening_outcomes.nh_cancers`', client)
    print("Done pulling cancers: {:2f}".format((time.time()-t0)/60))
    t0 = time.time()
    if limit is not None:
        cancers_bq = cancers_bq.loc[cancers_bq.person_id < limit]
    cancers = np.zeros((num_people, globs.numCancersAllowed, globs.num_cancer_id))
    # This is way faster than people and polyps query (most people don't have cancer), so probably don't need to speed up at all
    for i, row in cancers_bq.iterrows():
        person_id = int(row.person_id)
        cancer_id = int(row.cancer_id)
        cancers[person_id,cancer_id,globs.indLocationCancer] = row.Location
        cancers[person_id,cancer_id,globs.indAgeCancerInit] = row.AgeInit
        cancers[person_id,cancer_id,globs.indCancerGrowthRate] = row.GrowthRate
        cancers[person_id,cancer_id,globs.indSojournTCancer] = row.SojournTime
        cancers[person_id,cancer_id,globs.indSizeAtSymptoms] = row.SizeAtSymptoms
        cancers[person_id,cancer_id,globs.indStageAtSymptoms] = row.StageAtSymptoms
        cancers[person_id,cancer_id,globs.indAgeAtSymptoms] = row.AgeAtSymptoms
        cancers[person_id,cancer_id,globs.indSurvivalYearsSymptoms] = row.SurvivalYearsSymptoms
    print("Done setting up cancers: {:2f}".format((time.time()-t0)/60))
    return cancers


def make_results_df(people, costs, screener):
    '''Makes a dataframe of perperson results given arrays of simulated people, costs over time, and the screener
    people -- people array
    costs -- costs over time array
    screener -- name of screener. 'new technology' or 'cancer' or 'FIT' or 'cologuard' or 'none'
    Returns two dataframes. One for people results, the other for costs over time
    '''

    results = {}
    ### aggregate into df
    for i, person in enumerate(people):
        person_result = {}
        person_result['person_id'] = i
        if person[globs.indAgeCancerDeath] > 0:
            person_result['Cancer_Death_Age'] = person[globs.indAgeCancerDeath]
        # determine cause of death
        if person[globs.indAgeComplicationDeath] > 0:
            person_result['Death_Cause'] = 'complication'
            person_result['Death_Age'] = person[globs.indAgeComplicationDeath]
        elif (person[globs.indAgeOtherDeath] > person[globs.indAgeCancerDeath]) & (person[globs.indAgeCancerDeath] > 0):
            # NOTE: if (rounded) cancer death and other death are the same year, then we call it other death
            person_result['Death_Cause'] = 'cancer'
            person_result['Death_Age'] = person[globs.indAgeCancerDeath]
        else:
            person_result['Death_Cause'] = 'other'
            person_result['Death_Age'] = person[globs.indAgeOtherDeath]

        if person[globs.indAgeDiagnosed] > 0:
            person_result['Diag_Age'] = person[globs.indAgeDiagnosed]
            person_result['Diag_Stage'] = person[globs.indStageDiagnosed]
            person_result['Diag_Location'] = person[globs.indLocDiagnosed]
            person_result['Diag_Reason'] = 'Symptoms' if person[globs.indReasonDiagnosed] == globs.diagSymptoms else 'Intervention'
            
        person_result['Costs_Screen'] = person[globs.indCostScreen]
        person_result['Costs_Treatment'] = person[globs.indCostTreatment]
        person_result['Costs_Colos'] = person[globs.indCostColonoscopy]
        person_result['Costs_Comp'] = person[globs.indCostComplications]
        person_result['Costs'] = person[globs.indCostScreen] + person[globs.indCostTreatment] + person[globs.indCostColonoscopy] + person[globs.indCostComplications]

        person_result['NumScreens'] = person[globs.indNumScreens]
        person_result['NumColos'] = person[globs.indNumDiagColos] + person[globs.indNumSurveillanceColos] + person[globs.indNumSymptColos]
        if screener == 'colonoscopy' or (screener == 'new technology' and globs.SCREEN_KIND_NT == globs.valInvasive):
            person_result['NumColos'] += person[globs.indNumScreens]
        person_result['NumDiagColos'] = person[globs.indNumDiagColos]
        person_result['NumSurvColos'] = person[globs.indNumSurveillanceColos]
        person_result['NumSymptColos'] = person[globs.indNumSymptColos]

        person_result['QALY'] = person_result['Death_Age'] + person[globs.indLYAdjustment]

        results[i] = person_result

    results = pd.DataFrame.from_dict(results, orient='index')

    costs_year_df = {}
    for y in range(globs.MAX_AGE):
        costs_year_df[y] = {
            'year':y, 
            'cost':costs[y]
        }
    costs_year_df = pd.DataFrame.from_dict(costs_year_df, orient='index')

    return results, costs_year_df


def setup_globs(screener, extra_pars=None):
    '''Overwrites new technology screener variables in globs
    screener -- {dict} Parameters of the new techonology screener
    extra_pars -- {dict} Any additional parameters for the simulation
    '''
    # idea: if we would like the option to change SOC at all, pass in optional dicts with alternate SOC formulations
    
    globs.INTERVAL_NT, globs.ADHERENCE_NT = screener['interval'], screener['adherence']
    globs.SENS_SMALL_NT, globs.SENS_MED_NT, globs.SENS_LARGE_NT = screener['sensitivity_small'], screener['sensitivity_med'], screener['sensitivity_large']
    globs.SENS_CANCER_NT, globs.SPEC_NT = screener['sensitivity_cancer'], screener['specificity']
    globs.COST_NT = screener['cost']
    globs.TYPE_SCREEN_NT = globs.valInvasive if screener['kind'] == 'invasive' else globs.valNonInvasive
    globs.SCREENING_AGE_MIN_NT = screener['screening_age_min']
    globs.SCREENING_AGE_MAX_NT = screener['screening_age_max']
    
    # NOTE: there is no point in doing this right now, as only SOC simulations with period life tables are held in BQ
    if extra_pars is not None:
        globs.LIFE_TABLE_TYPE = extra_pars['life_table']
    if globs.LIFE_TABLE_TYPE == 'cohort':
        globs.BIRTH_YEAR = 1980
    else:
        globs.BIRTH_YEAR = 1975
    return


def run_preload(creds, n_people, n_jobs=1):
    '''Runs natural history and SOC screeners, then uploads results to bigquery
    creds -- The client credentials for gcp (NOTE: NOT the client, just the credentials)
    n_people -- {int} The number of people to simulate
    n_jobs -- {int} The number of processes to run asynchronously
    '''

    # Run a small sim so as to compile jitted functions. If we don't do this, each subprocess will do its own compilation when we run the full batch (which is fine, but eats more resources)
    res = run(int(1e1), n_jobs=1)
    # Run the full set of simulations
    res = run(n_people, n_jobs=n_jobs)
    # Unpack all the results
    people, polyps, cancers = res[0], res[1], res[2]
    results_none, results_costs_none, results_colo, results_costs_colo = res[3], res[4], res[5], res[6]
    results_cg, results_costs_cg, results_fit, results_costs_fit, _, _ = res[7], res[8], res[9], res[10], res[11], res[12]
    results = [results_none, results_colo, results_cg, results_fit]
    results_costs = [results_costs_none, results_costs_colo, results_costs_cg, results_costs_fit]
    
    # change dtypes to place nicely with bigquery
    for name, df, df_costs in zip(['none','colonoscopy','cologuard','fit'],results, results_costs):
        df['Diag_Reason'] = df.Diag_Reason.replace(np.nan, '')
        df['Screener'] = name
        df_costs['Screener'] = name
        df = df.astype({
            'Death_Cause': 'str',
            'NumScreens':'int', 
            'NumColos':'int', 
            'NumDiagColos':'int', 
            'NumSurvColos':'int',
            'NumSymptColos':'int',
            'person_id':'int',
            'Diag_Reason':'str'
        })
    results_all = results_none
    results_all = results_all.append(results_colo)
    results_all = results_all.append(results_cg)
    results_all = results_all.append(results_fit)
    results_costs_all = results_costs_none
    results_costs_all = results_costs_all.append(results_costs_colo)
    results_costs_all = results_costs_all.append(results_costs_cg)
    results_costs_all = results_costs_all.append(results_costs_fit)
    
    
    # convert people array into a reduced dataframe
    df_people = {}
    for i, person in enumerate(people):
        temp = {}
        temp['pid'] = i
        temp['byear'] = globs.BIRTH_YEAR
        temp['Gender'] = 'Male' if person[globs.indGender] == globs.genderMale else 'Female'
        temp['AgeOtherDeath'] = person[globs.indAgeOtherDeath]
        temp['PolypRisk'] = person[globs.indPolypRisk]
        df_people[i] = temp
    df_people = pd.DataFrame.from_dict(df_people, orient='index')

    # convert polyps array into a reduced dataframe
    df_polyps = {}
    counter = 0
    for i, person in enumerate(polyps):
        for p, polyp in enumerate(polyps[i]):
            if polyp[globs.indAgeInitPolyp] == 0:
                # row was never a polyp. Do not store
                continue
            temp = {}
            temp['person_id'] = i
            temp['polyp_id'] = p
            temp['Location'] = polyp[globs.indLocationPolyp]
            temp['AgeInit'] = polyp[globs.indAgeInitPolyp]
            temp['GrowthRate'] = polyp[globs.indGrowthRatePolyp]
            temp['SizeAtCancer'] = polyp[globs.indSizeAtCancer]
            temp['cancer_id'] = polyp[globs.indIsCancer]
            temp['AgeAtTransition'] = polyp[globs.indAgeAtTransition]
        
            df_polyps[counter] = temp
            counter += 1
    df_polyps = pd.DataFrame.from_dict(df_polyps, orient='index')

    # convert cancers array into reduced dataframe
    df_cancers = {}
    counter = 0
    for i, person in enumerate(cancers):
        for c, cancer in enumerate(cancers[i]):
            if cancer[globs.indAgeCancerInit] == 0:
                # row was never a cancer. Do not store
                continue
            temp = {}
            temp['person_id'] = i
            temp['cancer_id'] = c
            temp['Location'] = cancer[globs.indLocationCancer]
            temp['AgeInit'] = cancer[globs.indAgeCancerInit]
            temp['GrowthRate'] = cancer[globs.indCancerGrowthRate]
            temp['SojournTime'] = cancer[globs.indSojournTCancer]
            temp['SizeAtSymptoms'] = cancer[globs.indSizeAtSymptoms]
            temp['StageAtSymptoms'] = cancer[globs.indStageAtSymptoms]
            temp['AgeAtSymptoms'] = cancer[globs.indAgeAtSymptoms]
            temp['SurvivalYearsSymptoms'] = cancer[globs.indSurvivalYearsSymptoms]
            
            df_cancers[counter] = temp
            counter += 1
    df_cancers = pd.DataFrame.from_dict(df_cancers, orient='index')

    # Upload all results
    project_name = '[server]'
    pandas_gbq.to_gbq(df_people,'rdt_crc_screening_outcomes.nh_people',project_id=project_name, if_exists='replace', credentials=creds)
    pandas_gbq.to_gbq(df_polyps,'rdt_crc_screening_outcomes.nh_polyps',project_id=project_name, if_exists='replace', credentials=creds)
    pandas_gbq.to_gbq(df_cancers,'rdt_crc_screening_outcomes.nh_cancers',project_id=project_name, if_exists='replace', credentials=creds)
    pandas_gbq.to_gbq(results_all, 'rdt_crc_screening_outcomes.SOC_results', project_id=project_name, if_exists='replace', credentials=creds)
    pandas_gbq.to_gbq(results_costs_all, 'rdt_crc_screening_outcomes.SOC_results_costs', project_id=project_name, if_exists='replace', credentials=creds)
    return


def run(n_people, n_jobs=1):
    '''Setup of parameters, runs natural history, then overlays colonoscopy and cologuard and new technology screening.
    Simulates for n individuals
    Right now, everything is hardcoded and not generalized. Parameters are set as globals
    Returns dataframes of results summary for each screening scenario
    '''

    tr = time.time() #for timing
    '''
    new_tech = {#colonoscopy
        'kind': 'noninvasive',
        'interval':10.0, 
        'screening_age_min':50,
        'screening_age_max':80,
        'adherence': 1.0, 
        'sensitivity_small': 0.75, 
        'sensitivity_med': 0.85, 
        'sensitivity_large': 0.95, 
        'sensitivity_cancer': 0.95,
        'specificity': 0.86,
        'cost': 1337.46
        }
    '''
    new_tech = {#cologuard
        'kind': 'noninvasive',
        'interval':3.0, 
        'screening_age_min':50,
        'screening_age_max':80,
        'adherence': 1.0, 
        'sensitivity_small': 0.172, 
        'sensitivity_med': 0.172, 
        'sensitivity_large': 0.424, 
        'sensitivity_cancer': 0.923,
        'specificity': 0.898,
        'cost': 508.87
        }
    
    extra_pars = {
        'life_table':'cohort'
    }
    globs.BIRTH_YEAR = 1980

    setup_globs(new_tech, extra_pars=extra_pars)

    # TODO: ADD ADHERENCE TO SCREENING ROUTINES
    people = np.zeros((n_people,globs.num_person_id))
    polyps = np.zeros((n_people,globs.numPolypsAllowed,globs.num_polyp_id))
    cancers = np.zeros((n_people,globs.numCancersAllowed,globs.num_cancer_id))

    ## Filling in values ##

    people[:,globs.indGender] = np.random.rand(n_people)
    # Data location: LungCancer/USA_lifetables/STATS/Births.txt [from big dowloaded life table tarball from USCB, I think]
    # FROM census (births by gender/year): 
    # 1980: F/1832578   M/1927983   Total/3760561
    # 1975 F/1531063    M/1613135   Total/3144198
    if globs.BIRTH_YEAR == 1975:
        prop_male = 1613135.0/3144198.0  #.51305
    elif globs.BIRTH_YEAR == 1980:
        prop_male = 1927983.0/3760561.0 #.51268494
    else:
        raise ValueError("Birth year {} is not implemented".format(globs.BIRTH_YEAR))
    males, females = people[:,globs.indGender] < prop_male, people[:,globs.indGender] >= prop_male
    people[males,globs.indGender] = globs.genderMale
    people[females,globs.indGender] = globs.genderFemale

    # Other-cause mortality
    # Life table for 1980 cohort, given as conditional probabilities of death
    if globs.LIFE_TABLE_TYPE == 'cohort':
        globs.MAX_AGE = 120
        if globs.BIRTH_YEAR == 1980:
            table_name = '1980_cohort.csv'
        else:
            raise NotImplementedError("Cohort lifetable not implemented for birth year {}".format(globs.BIRTH_YEAR))
    elif globs.LIFE_TABLE_TYPE == 'period':
        table_name = '2012_period.csv'
    else:
        raise ValueError("life table type {} not supported".format(globs.LIFE_TABLE_TYPE))
    life_table = pd.read_csv(table_name)
    life_table = life_table[['Age','M','F']].values
    life_table[:,1:] = 1-np.cumprod(1-life_table[:,1:], axis=0)
    life_table[-1,1:] = 1
    # determine years of other-cause death for each person
    people = get_other_death_dates(people, life_table)
    people[:,globs.indAge] = 0.0
    people[:,globs.indAlive] = 1.0
    # Determine individual polyp risk for each person
    people[:,globs.indPolypRisk] = np.random.normal(loc=np.ones(n_people)*globs.POLYP_GEN_ALPHA_0, scale=np.ones(n_people)*globs.POLYP_GEN_SIGMA_ALPHA)

    ##############################
    ## SIMULATE NATURAL HISTORY
    ##############################
    people, polyps, cancers = natural_history(people, polyps, cancers)
    
    ##############################
    ## SIMULATE NO SCREENING
    ##############################
    people_none = people.copy()
    polyps_none = polyps.copy()
    cancers_none = cancers.copy()
    costs_none = np.zeros(globs.MAX_AGE)

    #people_none[:,globs.indAge] = 0
    #people_none[:, globs.indAlive] = 1
    people_none, polyps_none, cancers_none, costs_none = run_screening(people_none, polyps_none, cancers_none, costs_none, "none", n_jobs=n_jobs)

    results_none, results_costs_none = make_results_df(people_none, costs_none, 'none')

    ##############################
    ## SIMULATE COLONOSCOPY SCREENING
    ##############################
    people_colo = people.copy()
    polyps_colo = polyps.copy()
    cancers_colo = cancers.copy()
    costs_colo = np.zeros(globs.MAX_AGE)

    #people_colo[:,globs.indAge] = 0
    #people_colo[:, globs.indAlive] = 1
    people_colo, polyps_colo, cancers_colo, costs_colo = run_screening(people_colo, polyps_colo, cancers_colo, costs_colo, "colonoscopy", n_jobs=n_jobs)

    results_colo, results_costs_colo = make_results_df(people_colo, costs_colo, 'colonoscopy')
    
    ##############################
    ## SIMULATE COLOGUARD SCREENING
    ##############################
    people_cg = people.copy()
    polyps_cg = polyps.copy()
    cancers_cg = cancers.copy()
    costs_cg = np.zeros(globs.MAX_AGE)

    #people_cg[:,globs.indAge] = 0
    #people_cg[:, globs.indAlive] = 1
    '''
    screener = {
        'kind': 'stool',
        'interval':0.0, 
        'screening_age_min':81,
        'screening_age_max':80,
        'adherence': 1.0, 
        'sensitivity_small': 0.172, 
        'sensitivity_med': 0.172, 
        'sensitivity_large': 0.424, 
        'sensitivity_cancer': 0.923,
        'specificity': 0.898,
        'cost': 508.87
        }
    setup_globs(screener)
    '''
    people_cg, polyps_cg, cancers_cg, costs_cg = run_screening(people_cg, polyps_cg, cancers_cg, costs_cg, "cologuard", n_jobs=n_jobs)

    results_cg, results_costs_cg = make_results_df(people_cg, costs_cg, 'cologuard')

    ##############################
    ## SIMULATE FIT SCREENING
    ##############################
    people_fit = people.copy()
    polyps_fit = polyps.copy()
    cancers_fit = cancers.copy()
    costs_fit = np.zeros(globs.MAX_AGE)

    people_fit, polyps_fit, cancers_fit, costs_fit = run_screening(people_fit, polyps_fit, cancers_fit, costs_fit, 'FIT', n_jobs=n_jobs)

    results_fit, results_costs_fit = make_results_df(people_fit, costs_fit, 'FIT')
    
    ##############################
    ## SIMULATE NEW TECH SCREENING
    ##############################
    people_nt = people.copy()
    polyps_nt = polyps.copy()
    cancers_nt = cancers.copy()
    costs_nt = np.zeros(globs.MAX_AGE)

    # NOTE: we should be able to delete these, and all other instances other than the ones in run-screening-helper_inner
    #people_nt[:,globs.indAge] = 0
    #people_nt[:, globs.indAlive] = 1
    people_nt, polyps_nt, cancers_nt, costs_nt = run_screening(people_nt, polyps_nt, cancers_nt, costs_nt, "new technology", n_jobs=n_jobs)

    results_new, results_costs_new = make_results_df(people_nt, costs_nt, 'new technology')
    
    #results_cg, results_costs_cg = 0,0
    #results_fit, results_costs_fit = 0,0
    #results_new, results_costs_new = 0,0

    return people, polyps, cancers, results_none, results_costs_none, results_colo, results_costs_colo, results_cg, results_costs_cg, results_fit, results_costs_fit, results_new, results_costs_new


@numba.njit
def get_other_death_dates(people, dist):
    '''Determines and sets the ages at which each person will die from other causes, given a discrete cdf
    '''
    for _, person in enumerate(people):
        x = np.random.random()
        idx = 1 if person[globs.indGender] == globs.genderMale else 2
        for val in dist:
            if val[idx] >= x:
                age = val[0]
                break
        person[globs.indAgeOtherDeath] = age
    return people


def natural_history(people, polyps, cancers):
    ''' Runs and creates the natural history of each person, independent of any medical intervention of any kind. Multiple "parallel universes" are
    simulated, meaning even if a person would otherwise have gotten and died from cancer, we keep simulating progression of other adenomas/carcinomas
    This allows a direct comparison of results at the individual level for multiple different screeners without the need to resimulate the natural history
    
    This routine cant be jitted because of the 'get_poisson_lambda' subroutine which cant be jitted due to lack of numba support for poisson random sampling.
    Also, get_symptom_size_probability cant be jitted as the stats module is not supported, but there is an easier workaround for that if needed
    '''

    # Assemble staging parameters into intended matrix (numba doesn't allow arrays as constants, so can't be initialized in globs)
    CANCER_STAGE_PARS = np.array(
        [[globs.csp11, globs.csp12, globs.csp13],
        [globs.csp21, globs.csp22, globs.csp23],
        [globs.csp31, globs.csp32, globs.csp33]])
    # Pre-compute the cdf of cancer size at symptoms
    unscaled_cdf = np.cumsum(np.array([get_symptom_size_probability(s) for s in range(1,141)]))
    CANCER_SIZE_CDF = np.zeros((140,2))
    CANCER_SIZE_CDF[:,0] = range(1,141)
    CANCER_SIZE_CDF[:,1] = unscaled_cdf / unscaled_cdf[-1]

    idx_alive = people[:, globs.indAlive] == 1
    people_age = 0
    while np.any(idx_alive) and people_age < globs.MAX_AGE:

        if people_age < 20:
            # Adenomas do not develop before age 20
            # Determine who is still alive
            newDeaths = (people[:, globs.indAge] >= people[:, globs.indAgeOtherDeath]) & (people[:, globs.indAlive] == 1)
            people[newDeaths, globs.indAlive] = 0
            idx_alive = people[:, globs.indAlive] == 1
            people[idx_alive, globs.indAge] += 1
            people_age += 1
            continue

        # Each person that has polyps and is not dead will have a progression of their disease
        idx_has_polyps = (people[:,globs.indNumActivePolyps] > 0.5) & idx_alive
        people[idx_has_polyps], polyps[idx_has_polyps], cancers[idx_has_polyps] = progress_polyps(people[idx_has_polyps], polyps[idx_has_polyps], cancers[idx_has_polyps], people_age)
        
        # A cancer is new if it has an age_init == people_age (it was initiated this year)
        idx_has_new_cancers = idx_alive & np.any(cancers[:,:,globs.indAgeCancerInit] == people_age, axis=1) # people with at least one new developed cancer
        # For each person with new cancer, set each cancer up
        # It may just be worth doing the entire cancer setup in progress_polyps funcion (call it 'progress_disease' and do everything?)
        cancers[idx_has_new_cancers] = setup_cancers(people[idx_has_new_cancers], cancers[idx_has_new_cancers], CANCER_STAGE_PARS, CANCER_SIZE_CDF)#, cancer_size_cdf)

        # Each person that is not dead from other/noncancer causes has the chance to develop new polyps
        # Determine how many new polyps each person gets this year
        lambdas = get_poisson_lambda(people[idx_alive], people_age)
        num_new_polyps = np.zeros(len(people))
        num_new_polyps[idx_alive] = np.random.poisson(lambdas) # can't numba poisson (meaning we can't numba this whole routine :( -- Not worth it to write own method to sample from poisson from unif or exp variates)

        idx_gets_polyps = num_new_polyps >= 1 # indices of people who develop new polyps
        # For each person with new polyps, create new polyps, setup each new polyp
        people[idx_gets_polyps], polyps[idx_gets_polyps] = setup_polyps(
            people[idx_gets_polyps], polyps[idx_gets_polyps], num_new_polyps[idx_gets_polyps])
        
        # A person may die of other/noncancer causes
        newDeaths = (people[:, globs.indAge] >= people[:, globs.indAgeOtherDeath]) & (people[:, globs.indAlive] == 1)
        people[newDeaths, globs.indAlive] = 0

        # End of step

        # Determine who is still alive
        idx_alive = people[:, globs.indAlive] == 1
        # Everyone still alive gets one year older
        people[idx_alive, globs.indAge] += 1
        people_age += 1

    return people, polyps, cancers


@numba.njit
def get_poisson_lambda(people, people_age):
    '''Returns the lambda/rate parameter for poisson ademona generation
    people -- {array} The people array, subsetted to those that are still simulating
    people_age -- {numeric} The current age of people
    returns {array} The average lambdas of the nonhomogeneous poisson distribution per person
    '''
    
    if people_age < 45:
        age_bracket = 0
    elif people_age < 65:
        age_bracket = 1
    elif people_age < 75:
        age_bracket = 2
    else:
        age_bracket = 3

    risks = people[:,globs.indPolypRisk] #ALPHA_0_i
    gender_risks = people[:,globs.indGender] * globs.POLYP_GEN_ALPHA_1 # gender*ALPHA_1
    if people_age < 45:
        age_risks = people[:,globs.indAge] * globs.POLYP_GEN_ALPHA_2K_45#[age_bracket]
        extra_age_risks = globs.POLYP_GEN_SUM_AJDIFF_45
    elif people_age < 65:
        age_risks = people[:,globs.indAge] * globs.POLYP_GEN_ALPHA_2K_65
        extra_age_risks = globs.POLYP_GEN_SUM_AJDIFF_65
    elif people_age < 75:
        age_risks = people[:,globs.indAge] * globs.POLYP_GEN_ALPHA_2K_75
        extra_age_risks = globs.POLYP_GEN_SUM_AJDIFF_75
    else:
        age_risks = people[:,globs.indAge] * globs.POLYP_GEN_ALPHA_2K_120
        extra_age_risks = globs.POLYP_GEN_SUM_AJDIFF_120

    #lambda_i = ALPHA_0_i + gender*ALPHA_1 + age*ALPHA_2K[age_bracket] + SUM_AJdiff[age_bracket]
    lambdas = risks + gender_risks
    lambdas += age_risks
    lambdas += extra_age_risks
    lambdas = np.exp(lambdas)

    return lambdas


@numba.njit
def setup_polyps(people, polyps, num_new_polyps):
    '''Given people the number of new polyps a person should develop this time step, create instances of each polyp
    people -- the people array, subsetted to only people who get at least one new polyp this step
    polyps -- the polyp array, also subsetted
    num_new_polyps -- {array} The number of polyps each person (who is developing >= 1) is developing this step
    '''
    
    for i, person in enumerate(people):
        num_ever_polyps = np.sum(polyps[i,:,globs.indAgeInitPolyp] > 0)
        if num_ever_polyps == polyps.shape[1]:
            # person already has max number of polyps. Don't create any new ones
            continue

        for p in range(int(num_new_polyps[i])):
            polyp_id = num_ever_polyps + p # if there were n polyps before, the next p should be at idx (n+p-1) [and p starts counting up from 0]
            if polyp_id > polyps.shape[1] - 1:
                # has hit max number of polyps already. Don't create any new ones
                continue
            
            # mark this polyp as active
            polyps[i, polyp_id, globs.indIsActive] = 1
            person[globs.indNumActivePolyps] += 1
            # age of initiation of this polyp is the persons current age
            polyps[i, polyp_id, globs.indAgeInitPolyp] = person[globs.indAge]
            
            # determine polyp location
            x = np.random.random()
            if x < .09:
                # P(rectum) -> 0.09
                location = globs.locRectum # rectum
            elif x < .33:
                # P(sigmoid) -> 0.24
                location = globs.locSigmoid # sigmoid
            elif x < .45:
                # P(descending) -> 0.12
                location = globs.locDescending  # descending
            elif x < .69:
                # P(transverse) -> 0.24
                location = globs.locTransverse  # transverse
            elif x < .92:
                # P(ascending) -> 0.23
                location = globs.locAscending  # ascending
            else:
                # P(cecum) -> 0.08
                location = globs.locCecum # cecum
            polyps[i, polyp_id, globs.indLocationPolyp] = location

            # determine time for polyp to grow to 10mm
            if location == globs.locRectum:
                # rectum
                b1 = globs.polyp_growth_rbeta1
                b2 = globs.polyp_growth_rbeta2
            else:
                # anywhere in colon
                b1 = globs.polyp_growth_cbeta1
                b2 = globs.polyp_growth_cbeta2
            
            t10mm = b1 * (-np.log(np.random.random())) ** (-1/b2) # time to 10mm. This is the inverse cdf given in the docs (line 249)

            # determine growth rate of this polyp
            dmax = 50.0
            dmin = 1.0
            growth_rate = -np.log((dmax - 10)/(dmax - dmin)) / t10mm # line 254
            polyps[i, polyp_id, globs.indGrowthRatePolyp] = growth_rate

            # determine size at preclinical-crc transition/carcinoma creation
            if person[globs.indGender] == globs.genderFemale: 
                # female
                if location == globs.locRectum: 
                    #rectum
                    gamma1, gamma2 = .0470747, .0161731
                else: 
                    # colon
                    gamma1, gamma2 = .0444762, .0089362
            else:
                # male
                if location == globs.locRectum:
                    #rectum
                    gamma1, gamma2 = .0472322, .0173598
                else: 
                    # colon
                    gamma1, gamma2 = .04, .0089232
            gamma3 = .5
            mean = -np.log(gamma1) - gamma2*(polyps[i,polyp_id,globs.indAgeInitPolyp] - 50)
            
            # crc-aim uses a cycle-based approach, and thus uses probability of transition within each interval/time-step
            # conditioned on no transition in the past. It is more efficient to generate the size at cancer in advance
            # NOTE: see note in progress_polyps()
            #size = 0
            #while size > dmax or size < dmin:
            #    size = np.random.lognormal(mean, gamma3)
            size = np.random.lognormal(mean, gamma3)
            #globs.indCumProbCancer # the cumulative probability of having transitioned to cancer. For speedup
            #globs.indProbTransitionDistMean # the mean of lognormal dist for transition to cancer. For speedup
            #polyps[i, polyp_id, globs.indProbTransitionDistMean] = mean
            #polyps[i, polyp_id, globs.indCumProbCancer] = 0 #should already be the case though
            polyps[i, polyp_id, globs.indSizeAtCancer] = size
    
    return people, polyps


@numba.njit
def progress_polyps(people, polyps, cancers, people_age):
    '''Ages/progresses all active polyps for all alive people with active polyps.
    Rather than storing the current size, we just determine if the new size is big enough such that it becomes cancer
    If it does become cancer, create a new cancer (but don't fully set it up), and link the polyp to that cancer, and mark the polyp as inactive (it is now cancer)
    '''
    
    for person_idx, person in enumerate(people):
        for polyp_idx, polyp in enumerate(polyps[person_idx]):            
            if polyp[globs.indIsActive] == 0:
                # adenoma has become a carcinoma or has not been created, so go to next polyp
                continue 

            rate, t1 = polyp[globs.indGrowthRatePolyp], (people_age - polyp[globs.indAgeInitPolyp]) + 1 # age of polyp is people-age - age-at-polyp-init, so t1 is that + 1
            d_0, d_inf = 1.0, 50.0
            # Determine new polyp size by the end of this timestep (at t+1)
            d_t1 = d_inf - (d_inf - d_0) * np.exp(-rate * t1) # line 256

            #def xi(s,a):
            #    gamma1, gamma2, gamma3 = .04, .0089232, .5
            #    return norm.cdf((np.log(s*gamma1) + gamma2*(a-50))/gamma3)
            # NOTE: in crcaim, they don't take this equivalent shortcut.
            # crcaim computes the probability of transition in the next year conditional on it not having happened yet.


            # If polyp grows to a size bigger than the pre-sampled size at preclinical CRC transition, polyp becomes a cancer
            if d_t1 >= polyp[globs.indSizeAtCancer]:
                # Mark polyp as inactive
                polyp[globs.indIsActive] = 0
                person[globs.indNumActivePolyps] -= 1
                person[globs.indNumCancers] += 1
                if person[globs.indNumCancers] >= cancers.shape[1]: #reached max num of cancers
                    # we don't want to model a new cancer, but for now we will keep track of number of cancers beyond the max, to see if we need to adjust the maximum allowed
                    break

                # need to idx for new cancer.
                cancer_idx = int(person[globs.indNumCancers] - 1) # we subtract because we have already incremented numcancers.
                polyps[person_idx, polyp_idx, globs.indIsCancer] = cancer_idx + 1 # indicates this polyp has a cancer at indx polyp[indIsCancer]-1 in cancer array, and polyp doesn't need to be updated anymore
                polyps[person_idx, polyp_idx, globs.indAgeAtTransition] = people_age # the age at transition is rounded to the integer age of tthe person at the beginning of this time step
                cancers[person_idx, cancer_idx, globs.indLocationCancer] = polyp[globs.indLocationPolyp] # cancer has same location as polyp
                cancers[person_idx, cancer_idx, globs.indAgeCancerInit] = people_age


    return people, polyps, cancers


@numba.njit
def setup_cancers(people, cancers, cancer_stage_pars, cancer_size_cdf):
    '''Given people with new cancers and the cancers which have been "initialized" but not "set up", determine key variables for each cancer
    people -- people array, subsetted to those with new cancers
    cancers -- cancer array, subsetted
    cancer_stage_pars -- array, the parameters used for sampling cancer stage given size
    cancer_size_cdf -- array, the cdf of the generalized log distribution used to determine cancer size at symptoms
    '''

    for idx_person, person in enumerate(cancers):
        for idx_cancer, cancer in enumerate(person):
            if (cancer[globs.indSojournTCancer] > 0) | (cancer[globs.indAgeCancerInit] == 0):
                # either (not a new cancer), or is an old cancer
                continue
            # Determine sojourn time
            if cancer[globs.indLocationCancer] == globs.locRectum:
                # rectum
                xi, nu = 1.148838, 0.564791
            else:
                # colon
                xi, nu = 0.943347, 0.492673
            #sojourn_time = np.random.lognormal(xi, nu) - .5
            sojourn_time = np.exp(xi + nu*np.random.normal()) # equivalent to np.random.lognormal(xi, nu) (I've double checked with experiments)
            cancer[globs.indSojournTCancer] = sojourn_time
            # Determine size at symptoms/clinical diagnosis
            size = get_cancer_size_from_ecdf_gen_log(cancer_size_cdf)
            cancer[globs.indSizeAtSymptoms] = size
            # Determine cancer growth rate
            cancer[globs.indCancerGrowthRate] = (2*size)**(1/sojourn_time) # 2*size is equivalent to size/.5 = size/min_size
            # Determine cancer stage at clinical detection
            cancer[globs.indStageAtSymptoms] = get_cancer_stage_given_size(size, cancer_stage_pars)
            # Determine persons age at clinical diagnosis
            cancer[globs.indAgeAtSymptoms] = people[idx_person, globs.indAge] + sojourn_time
            # Determine time from diagnosis to death from from this specific cancer, should that happen
            cancer[globs.indSurvivalYearsSymptoms] = get_death_from_cancer_age(people[idx_person,globs.indAge], people[idx_person,globs.indGender], cancer[globs.indLocationCancer], cancer[globs.indStageAtSymptoms])
            
    return cancers


def get_symptom_size_probability(s):
    '''Gets the probability density of a cancer exhibiting symptoms at size s
    s -- float, a size (diameter) in mm
    '''
    mu, sigma, lambd = 3.91048, 0.37775, 28.9135
    root = np.sqrt(s**2 + lambd**2)
    scaling = (s + root) / sigma
    scaling = scaling / (root**2 + s*root)
    normal_input = ( np.log((s + root)/2) - mu )/sigma
    prob = scaling * norm.pdf(normal_input)
    return prob


@numba.njit
def get_cancer_size_from_ecdf_gen_log(cancer_size_cdf):
    '''
    Given an array of the cdf sampled at a number of evenly spaced points, sample from the generalized log distribution
    cancer_size_cdf -- array, the cdf of gen log distribution
    '''

    x = np.random.random()
    size = cancer_size_cdf[0,0]
    for prob in cancer_size_cdf:
        if prob[1] > x:
            size = prob[0]
            break
        if prob[1] == 1:
            size = prob[0]
    return size


@numba.njit
def get_cancer_stage_given_size(c_size, cancer_stage_pars):
    ''' Randomly sample the stage of a cancer given the size in mm
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
        if location == globs.locRectum:
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
            shape = 0.699099 #This is 1/sigma == lambda. 
            intercept = 5.8797
            #NOTE: PAGE 91 DOES NOT INDICATE AN AGE EFFECT FOR 80+. WHERE DOES -1.7619 COME FROM?????
            # ANSWER: LINE 527 IN SUPPLEMENT
            age_effect = 1.609704 if age < 50 else 0.649916 if age < 60 else -0.011486 if age < 70 else -0.486255 if age < 80 else -1.7619
            # NOTE: gender is coded as +1 for female and -1 for males
            sex_effect = 0.13208 * (1 if gender == globs.genderFemale else -1)
            scale = intercept + age_effect + sex_effect # so this is correct
            # note that the quoted parameter sigma == 1/k, so we just use shape rather than 1/shape in the exponent here
            #NOTE: AGE/SEX EFFECTS ARE LINEARLY COMBINED WITH INTERCEPT REPRESENTETD BY LAMBDA IN THIS MODEL
            # THAT IS IN FACT WHAT I DID!
            # BUT WE NEED TO EXPONENTIATE BY 1/LAMBDA, SO I ASSUME WE EXPONENTIATE BY 1/(1/sigma + intercept + effects) = 1/(lambda + intercept + effects)
            time_from_diagnosis = scale * (-np.log(1-x))**(1/shape) #scale * (-np.log(1-x))**(shape)
    else:
        if stage == 2:
            if location == globs.locRectum:
                sigma = 2.1874
                intercept = 3.4680
                age_effect = 0.9608 if age < 50 else 0.4976 if age < 60 else 0.1146 if age < 70 else -0.3914 if age < 80 else -1.1817
                sex_effect = 0
            else:
                sigma = 2.8316
                intercept = 4.5013
                age_effect = 0.6193 if age < 50 else 0.3778 if age < 60 else 0.3243 if age < 70 else -0.2226 if age < 80 else -1.0987
                sex_effect = 0.1825 * (1 if gender == globs.genderFemale else -1)
        elif stage == 3:
            if location == globs.locRectum:
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
            if location == globs.locRectum:
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

    #NOTE: if sampled survival is longer than 100 years, we assume it is 100 years. Thats what crc-aim does. But basically makes no difference
    # NOTE: OR DOES IT? SHOULD IT BE RESAMPLED (REDISTRIBUTE DENSITY PAST 100 YEARS? I DOUBT IT, BUT MAYBE)
    # NOTE: adding +1 here (got rid of it)
    survival_years = min(time_from_diagnosis, 100) #+ 1
    return survival_years


def run_screening(people, polyps, cancers, costs, screen_name, n_jobs=1):
    '''Overlays a screening protocol on the prerun natural history simulation
    people -- The people array from the natural history simulation
    polyps -- The polyps array from the natural history simulation
    cancers -- The cancers array from the natural history simulation
    costs -- array(1xMax_age) Initially all zeros, will hold total population costs for each year/age
    screen_name -- {str} The name of the screener being simulated. If not one of the SOC screeners, the name must be "new technology"
        NOTE -- "new technology" is REQUIRED to be the name of the non SOC screener, or else this will not work
    n_jobs -- {int} The number of processes to run asynchronously
    Returns the people, polyps, cancers, and costs arrays with updated values
    '''

    if n_jobs > 1:
        # number people on each thread
        num_per_thread = len(people)//n_jobs 
        # start and end indices/ids of people to be run in each job
        slices = [(x*num_per_thread,(x+1)*num_per_thread) if x < n_jobs else (x*num_per_thread, len(people)) for x in range(n_jobs)]
        # make a copy of the slice of arrays to be sent to each subprocesses
        peoples = [people[sl[0]:sl[1]].copy() for sl in slices]
        polypss = [polyps[sl[0]:sl[1]].copy() for sl in slices]
        cancerss = [cancers[sl[0]:sl[1]].copy() for sl in slices]
        costss = [costs.copy() for sl in slices]

        P = mp.Pool(n_jobs)
        # run all processes
        results = [P.apply_async(run_screening_wrapper, 
            args=(
                peoples[x],
                polypss[x],
                cancerss[x],
                costss[x],
                screen_name,
                x+1)
            )
            for x in range(n_jobs)]
        P.close()
        P.join()
        # There is sometimes an issue retreiving the pickled subprocess due to the numba types not being pickleable?
        # So just the needed result data of each process is written to a pickle (named by process number), and that data is read back in, and file deleted
        # TODO: put these files in a different location?
        for x in range(n_jobs):
            with open('{}/people_{}.npy'.format(cwd,x+1), 'rb') as f:
                temp = np.load(f)
                people[slices[x][0]:slices[x][1]] = temp
            os.remove('{}/people_{}.npy'.format(cwd,x+1))
            
            with open('{}/polyps_{}.npy'.format(cwd,x+1), 'rb') as f:
                polyps[slices[x][0]:slices[x][1]] = np.load(f)
            os.remove('{}/polyps_{}.npy'.format(cwd,x+1))
            
            with open('{}/cancers_{}.npy'.format(cwd,x+1), 'rb') as f:
                cancers[slices[x][0]:slices[x][1]] = np.load(f)
            os.remove('{}/cancers_{}.npy'.format(cwd,x+1))
            
            with open('{}/costs_{}.npy'.format(cwd,x+1), 'rb') as f:
                costs += np.load(f)
            os.remove('{}/costs_{}.npy'.format(cwd,x+1))

    else:
        # Run only on current process
        results = run_screening_wrapper(people, polyps, cancers, costs, screen_name, 0)
        people, polyps, cancers, costs = results[0], results[1], results[2], results[3]

    return people, polyps, cancers, costs


def run_screening_wrapper(people, polyps, cancers, costs, screen_name, process_no):
    '''Spawns screening overlay on a subprocess and writes results to pickles
    people -- The people array from the natural history simulation, but with age and alive set back to defaults
    polyps -- The polyps array from the natural history simulation
    cancers -- The cancers array from the natural history simulation
    costs -- array(1xMax_age) Initially all zeros, will hold total population costs for each year/age
    screen_name -- {str} The name of the screener being simulated. If not one of the SOC screeners, the name must be "new technology"
        NOTE -- "new technology" is REQUIRED to be the name of the non SOC screener, or else this will not work
    process_no -- {int} The number/name of the subprocess (for pickle file naming purposes). 0 if not a subprocess
    Returns the people, polyps, cancers, and costs arrays with updated values
    '''
    
    people, polyps, cancers, costs = run_screening_simulate(people,polyps,cancers,costs,screen_name)
    if process_no != 0:
        with open('{}/people_{}.npy'.format(cwd,process_no), 'wb') as f:
            np.save(f, people)
        with open('{}/polyps_{}.npy'.format(cwd,process_no), 'wb') as f:
            np.save(f, polyps)
        with open('{}/cancers_{}.npy'.format(cwd,process_no), 'wb') as f:
            np.save(f, cancers)
        with open('{}/costs_{}.npy'.format(cwd,process_no), 'wb') as f:
            np.save(f, costs)
        return
    else:
        results = (people, polyps, cancers, costs)
        return results


@numba.njit
def run_screening_simulate(people, polyps, cancers, costs, screen_name):
    '''Overlays a screening protocol on the prerun natural history simulation
    people -- The people array from the natural history simulation, but with age and alive set back to defaults
    polyps -- The polyps array from the natural history simulation
    cancers -- The cancers array from the natural history simulation
    costs -- array(1xMax_age) Initially all zeros, will hold total population costs for each year/age
    screen_name -- {str} The name of the screener being simulated. If not one of the SOC screeners, the name must be "new technology"
        NOTE -- "new technology" is REQUIRED to be the name of the non SOC screener, or else this will not work
    Returns the people, polyps, cancers, and costs arrays with updated values
    '''
    
    # retrieve constants for the screening overlay
    CANCER_STAGE_PARS = np.array([[globs.csp11, globs.csp12, globs.csp13],[globs.csp21, globs.csp22, globs.csp23],[globs.csp31, globs.csp32, globs.csp33]])
    SCREENING_AGE_MIN = globs.SCREENING_AGE_MIN_COLO if screen_name == 'colonoscopy' else globs.SCREENING_AGE_MIN_CG if screen_name == 'cologuard' else globs.SCREENING_AGE_MIN_FIT if screen_name == 'FIT' else globs.SCREENING_AGE_MIN_NT
    SCREENING_AGE_MAX = globs.SCREENING_AGE_MAX_COLO if screen_name == 'colonoscopy' else globs.SCREENING_AGE_MAX_CG if screen_name == 'cologuard' else globs.SCREENING_AGE_MAX_FIT if screen_name == 'FIT' else globs.SCREENING_AGE_MAX_NT
    SCREEN_ADHERENCE = globs.ADHERENCE_COLO if screen_name == 'colonoscopy' else globs.ADHERENCE_CG if screen_name == 'cologuard' else globs.ADHERENCE_FIT if screen_name == 'FIT' else globs.ADHERENCE_NT
    SCREEN_INTERVAL = globs.INTERVAL_COLO if screen_name == 'colonoscopy' else globs.INTERVAL_CG if screen_name == 'cologuard' else globs.INTERVAL_FIT if screen_name == 'FIT' else globs.INTERVAL_NT
    SCREEN_TYPE = globs.TYPE_SCREEN_COLO if screen_name == 'colonoscopy' else globs.TYPE_SCREEN_CG if screen_name == 'cologuard' else globs.TYPE_SCREEN_FIT if screen_name == 'FIT' else globs.TYPE_SCREEN_NT if screen_name == 'new technology' else -1
    SCREEN_TYPE = 'invasive' if SCREEN_TYPE == globs.valInvasive else 'noninvasive' if SCREEN_TYPE == globs.valNonInvasive else 'none'

    # Reset base attributes
    people[:, globs.indAge] = 0
    people[:, globs.indAlive] = 1

    idx_alive = people[:, globs.indAlive] == 1
    people_age = 0
    
    while np.any(idx_alive) and people_age < globs.MAX_AGE:
        # We are assuming nobody can get diagnosed with cancer prior to age 20
        
        if people_age < 20:
            # Death by other causes
            newOtherDeaths = (
                idx_alive & 
                (people[:, globs.indAge] == people[:, globs.indAgeOtherDeath])
                )
            people[newOtherDeaths, globs.indAlive] = 0
            idx_alive = (people[:,globs.indAlive] == 1)
            people[idx_alive, globs.indAge] += 1
            people_age += 1
            continue
        '''
        # this is temporary. Want to look at count of adenomas at age 65
        if people_age == 65 and SCREEN_TYPE == 'none':
            # < 5mm, 6-9mm, 10+mm -> 16.2, 9.2, 18.4 per 1000 in RECTUM
            #polyps[idx_alive] # polyps of people alive at 65
            count_with = 0
            count_with2 = 0
            count_polyps = 0 # total number of polyps across people
            for ii, temp_person_polyps in enumerate(polyps[idx_alive]):
                #print((temp_person_polyps[:, globs.indAgeInitPolyp] < 65))
                #print((temp_person_polyps[:,globs.indIsCancer] == 0))
                #print(np.any((temp_person_polyps[:, globs.indAgeInitPolyp] < 65) & (temp_person_polyps[:,globs.indIsCancer] == 0)))
                #print((temp_person_polyps[:, globs.indAgeInitPolyp] < 65) & (temp_person_polyps[:,globs.indIsCancer] == 0))
                if np.any((temp_person_polyps[:, globs.indAgeInitPolyp] > 0) & (temp_person_polyps[:,globs.indIsCancer] == 0)):
                    count_with += 1
                #print(people[idx_alive][ii, globs.indAgeDiagnosed])
                if np.any((temp_person_polyps[:, globs.indAgeInitPolyp] > 0)) and (people[idx_alive][ii, globs.indAgeDiagnosed] == 0):
                    count_with2 += 1
                count_polyps += np.sum((temp_person_polyps[:, globs.indAgeInitPolyp] > 0) & (temp_person_polyps[:,globs.indIsCancer] == 0))
                
            print('polyp which is not cancer, no diagnosed restriction',count_with/np.sum(idx_alive))
            print('polyp which is not cancer, not diagnosed (with bug)',count_with/np.sum( idx_alive & (people[:,globs.indAgeDiagnosed] == 0) ))
            print('should be .2917 overall')
            print('mean polyps ', count_polyps/np.sum(idx_alive))
            #polyps[idx_alive,:,globs.indAgeInitPolyp] # want share of these that have any active polyps (right?) Or maybe not active? Should people no have diagnosed cancer?
        '''
        
        # Screening and surveillance should only be executed for those who are alive and have not had a cacner diagnosis
        idx_alive_not_diagnosed = (
            (people[:,globs.indAgeDiagnosed] == 0) &
            idx_alive
        )
        # Screening starts at SCREENING_AGE_MIN
        if (people_age == SCREENING_AGE_MIN) & (screen_name != "none"):
            people[idx_alive_not_diagnosed, globs.indAgeNextScreen] = people_age
        
        ## Sizes of lesions are updated
        for i, person in enumerate(people):
            if not idx_alive_not_diagnosed[i]:
                continue
            ppolyps = polyps[i]
            # Indices of existing polyps that have not progressed to cancer or have been resected
            idx_active_polyps = (
                ((ppolyps[:,globs.indAgeAtTransition] > person[globs.indAge]) | 
                (ppolyps[:,globs.indAgeAtTransition] == 0) ) &
                (ppolyps[:,globs.indAgeInitPolyp] <= person[globs.indAge]) & 
                (ppolyps[:,globs.indAgeInitPolyp] > 0))
            d_0, d_inf = 1.0, 50.0 # minimum and maximum polyp sizes
            # polyp age at each timestep
            t1s = person[globs.indAge] - ppolyps[idx_active_polyps,globs.indAgeInitPolyp] #NOTE: could add a one here? Timestep logic is ambiguous
            # NOTE: just modified above
            # Growth rates of each polyp
            rates = ppolyps[idx_active_polyps,globs.indGrowthRatePolyp]
            # Update polyp sizes for this timestep
            ppolyps[idx_active_polyps,globs.indSizePolyp] = d_inf - (d_inf - d_0) * np.exp(-rates * t1s)
            # all other polyps should have size 0
            ppolyps[~idx_active_polyps,globs.indSizePolyp] = 0

        ## Surveillance colonoscopies, if due
        idx_surveillance = (
            idx_alive_not_diagnosed & 
            (people[:, globs.indAgeNextSurvCol] == people_age)
            )

        # if using an "invasive" screen type, it is assumed to REPLACE the colonoscopy in all instances
        colo_type = screen_name if SCREEN_TYPE == 'invasive' else 'colonoscopy'
        people[idx_surveillance], polyps[idx_surveillance], cancers[idx_surveillance], costs[people_age] = get_surveillance_colos(people[idx_surveillance], polyps[idx_surveillance], cancers[idx_surveillance], costs[people_age], test_name=colo_type)
        
        ## Screening occurs, if due, but can't outside of screening window even if one was scheduled
        if people_age <= SCREENING_AGE_MAX and people_age >= SCREENING_AGE_MIN and screen_name != 'none':
            idx_screening = (
                idx_alive_not_diagnosed & 
                (people[:,globs.indAgeNextScreen] == people_age)
                )
            # assume screening adherence is independent of previous screening history
            idx_adherent = idx_screening & (np.random.random(len(idx_screening)) < SCREEN_ADHERENCE)
            idx_nonadherent = idx_screening & (~ idx_adherent)
            # nonadherent individuals will be assigned next screen in <interval> years
            people[idx_nonadherent, globs.indAgeNextScreen] += SCREEN_INTERVAL
            
            if SCREEN_TYPE == 'invasive':
                # could be a colonoscopy type screen, but still pass in name of screener, bc could use different metrics than regular colonoscopy
                people[idx_adherent], polyps[idx_adherent], cancers[idx_adherent], costs[people_age] = get_screened_colo(people[idx_adherent], polyps[idx_adherent], cancers[idx_adherent], costs[people_age], screen_name)
            else:
                people[idx_adherent], polyps[idx_adherent], cancers[idx_adherent], costs[people_age] = get_screened_blood_stool(people[idx_adherent], polyps[idx_adherent], cancers[idx_adherent], costs[people_age], screen_name)
            
        ## People may become symptomatic/receive clinical diagnosis
        idx_has_symptomatic_cancers = (people[:,globs.indAge] == -4000) # numba makes boolean array creation hard, so this does the trick
        # This is not vectorized because the axis argument in np.any() is not supported with numba which means that 
        # when a matrix is provided, it reduces to a scalar and not a vector/array, so a loop approach must be taken instead :(
        for i, pcancers in enumerate(cancers):
            idx_has_symptomatic_cancers[i] = (
                (people[i,globs.indAgeDiagnosed] == 0) & # agediagnosed is set right after this
                np.any( 
                    (pcancers[:,globs.indAgeCancerInit] > 0) & 
                    (pcancers[:,globs.indAgeCancerInit] <= people_age) & 
                    (pcancers[:,globs.indAgeAtSymptoms] <= people_age) 
                    )
            )
        idx_has_symptomatic_cancers = idx_has_symptomatic_cancers & idx_alive_not_diagnosed
        # people who are alive and get a symptomatic cancer this year are immediately diagnosed
        people[idx_has_symptomatic_cancers, globs.indAgeDiagnosed] = people_age
        if screen_name == 'new technology':
            if SCREEN_TYPE == 'invasive':#numba.literally(globs.TYPE_SCREEN_NT) == 'invasive':
                colo_type = 'new technology'
            else:
                colo_type = 'colonoscopy'
        else:
            colo_type = 'colonoscopy'
        # a colonoscopy is required to diagnose the cancer
        people[idx_has_symptomatic_cancers], cancers[idx_has_symptomatic_cancers], costs[people_age] = diagnose_cancer_symptomatic(people[idx_has_symptomatic_cancers], cancers[idx_has_symptomatic_cancers], costs[people_age], colo_type=colo_type)

        ## Diagnosis by screen
        idx_diagnosed_screen = (
            (people[:,globs.indAgeDiagnosed] == people_age) & 
            (~idx_has_symptomatic_cancers)
            )
        # those who were diagnosed this year but are not symptomatic must have been diagnosed via screening or surveillance
        people[idx_diagnosed_screen], cancers[idx_diagnosed_screen] = diagnose_cancer_colonoscopy(people[idx_diagnosed_screen], cancers[idx_diagnosed_screen], CANCER_STAGE_PARS)

        # Some people may have died from colonoscopy. Remove them from idx_alive before adding treatment costs
        idx_alive = (
            idx_alive &
            (people[:,globs.indAgeComplicationDeath] != people_age)
        )

        ## Add treatment costs
        idx_in_treatment = (
            (people[:,globs.indAgeDiagnosed] > 0) &
            idx_alive
        )
        people[idx_in_treatment], treatment_costs = add_treatment_costs(people[idx_in_treatment], people_age)
        costs[people_age] += treatment_costs

        ## Other cause death. Assumes ageotherdeath is integer
        newOtherDeaths = (
            idx_alive & 
            (people[:, globs.indAge] == people[:, globs.indAgeOtherDeath])
            )
        ## Cancer death
        newCancerDeaths = (
            idx_alive & 
            ((people[:,globs.indAgeDiagnosed] > 0) & (people[:, globs.indAge] == people[:, globs.indAgeCancerDeath])) &
            ~newOtherDeaths
            )
        ## Complication death
        newComplicationDeaths = (people[:,globs.indAgeComplicationDeath] == people_age)

        people[newComplicationDeaths, globs.indAlive] = 0
        people[newCancerDeaths, globs.indAlive] = 0
        people[newOtherDeaths, globs.indAlive] = 0

        ## End of step
        # adjust baseline LY by quality per age
        if people_age <= 24:
            people[idx_alive, globs.indLYAdjustment] -= 1 - globs.AGE_UTILITY_18_24
        elif people_age <= 34:
            people[idx_alive, globs.indLYAdjustment] -= 1 - globs.AGE_UTILITY_25_34
        elif people_age <= 44:
            people[idx_alive, globs.indLYAdjustment] -= 1 - globs.AGE_UTILITY_35_44
        elif people_age <= 54:
            people[idx_alive, globs.indLYAdjustment] -= 1 - globs.AGE_UTILITY_45_54
        elif people_age <= 64:
            people[idx_alive, globs.indLYAdjustment] -= 1 - globs.AGE_UTILITY_55_64
        elif people_age <= 74:
            people[idx_alive, globs.indLYAdjustment] -= 1 - globs.AGE_UTILITY_65_74
        else:
            people[idx_alive, globs.indLYAdjustment] -= 1 - globs.AGE_UTILITY_75

        idx_alive = (people[:, globs.indAlive] == 1)
        ## Get one year older
        people[idx_alive, globs.indAge] += 1
        people_age += 1
    return people, polyps, cancers, costs    


@numba.njit
def add_treatment_costs(people, people_age):
    '''Adds costs to people based on individual attributes and stage of diagnosis, and also adjust life-years based on quality
    
    Note that we assume state 0 = initial, 1 = continuing, 2 = terminal (cancer death), 3 = terminal (other death)
    Then have matrix C where C_ij is yearly cost for j = treatment phase, and i = (stage - 1)
    people -- people array, subsetted to those with diagnosed cancer
    people_age -- the age of the people
    returns the people array, and care_costs {float} which is the total amount spent on cancer treatment across all people this year
    '''

    CANCER_TREATMENT_COSTS = np.array(
        [[globs.CANCER_TREATMENT_COSTS_11, globs.CANCER_TREATMENT_COSTS_12, globs.CANCER_TREATMENT_COSTS_13, globs.CANCER_TREATMENT_COSTS_14],
        [globs.CANCER_TREATMENT_COSTS_21, globs.CANCER_TREATMENT_COSTS_22, globs.CANCER_TREATMENT_COSTS_23, globs.CANCER_TREATMENT_COSTS_24],
        [globs.CANCER_TREATMENT_COSTS_31, globs.CANCER_TREATMENT_COSTS_32, globs.CANCER_TREATMENT_COSTS_33, globs.CANCER_TREATMENT_COSTS_34],
        [globs.CANCER_TREATMENT_COSTS_41, globs.CANCER_TREATMENT_COSTS_42, globs.CANCER_TREATMENT_COSTS_43, globs.CANCER_TREATMENT_COSTS_44]])
    CANCER_UTILITIES = np.array(
        [[globs.TRMT_UTILITY_11, globs.TRMT_UTILITY_12, globs.TRMT_UTILITY_13, globs.TRMT_UTILITY_14],
        [globs.TRMT_UTILITY_21, globs.TRMT_UTILITY_22, globs.TRMT_UTILITY_23, globs.TRMT_UTILITY_24],
        [globs.TRMT_UTILITY_31, globs.TRMT_UTILITY_32, globs.TRMT_UTILITY_33, globs.TRMT_UTILITY_34],
        [globs.TRMT_UTILITY_41, globs.TRMT_UTILITY_42, globs.TRMT_UTILITY_43, globs.TRMT_UTILITY_44]])
    care_costs = 0
    for i, person in enumerate(people):
        # determine treatment stage
        death_cause = 'other' if person[globs.indAgeOtherDeath] <= person[globs.indAgeCancerDeath] else 'cancer' #if same year, assume other death
        death_age = person[globs.indAgeOtherDeath] if death_cause == 'other' else person[globs.indAgeCancerDeath]
        if death_age == people_age: #this is last year of life (remember),WE ARE ARE ASSUMING ALL DEATH AGES ARE INTEGER VALUED
            # terminal phase
            phase = 2 if death_cause == 'cancer' else 3            
        elif person[globs.indAgeDiagnosed] == people_age: # diagnosed this year, and they survive more than one year bc they failed above condition
            # initial phase
            phase = 0
        else:
            # continuing phase
            phase = 1
        stage = int(person[globs.indStageDiagnosed])
        care_costs += CANCER_TREATMENT_COSTS[stage-1,phase]
        person[globs.indCostTreatment] += CANCER_TREATMENT_COSTS[stage-1,phase]
        person[globs.indLYAdjustment] += CANCER_UTILITIES[stage-1,phase]
    return people, care_costs


@numba.njit
def get_surveillance_colos(people, polyps, cancers, ycosts, test_name='colonoscopy'):
    '''Surveillance Colonoscopies
    people -- people array, subsetted to those getting surveillance colonoscopies
    polyps -- polyps array, subsetted
    cancers -- cancers array, subsetted
    ycosts -- {float} The population costs so far this year
    Returns:
        people -- the updated people array
        polyps -- the updated polyps array
        cancers -- the updated cancers array
        ycosts -- the updated sum of costs
    '''
    costs_per_person = np.zeros(len(people))
    # NOTE: prange can be used to parallelize this function, but we don't do that any more due to server size, so this is legacy
    for i in numba.prange(len(people)):
        people[i], polyps[i], cancers[i], costs_per_person[i] = get_colonoscopy(people[i], polyps[i], cancers[i], 0, 'surveillance', False, test_name=test_name)
    ycosts += np.sum(costs_per_person)
    return people, polyps, cancers, ycosts


@numba.njit
def get_colonoscopy(person, ppolyps, pcancers, ycosts, kind, repeat, test_name='colonoscopy', prev_result=0):
    ''' A person gets a colonoscopy (of any type)
    If kind == 'symptom', then ppolyps and pcancers aren't needed, so they just need to be arbitrary arrays
    If repeat == True, this colo comes immediately after a previous colo (of any type) that did not achieve full reach
    NOTE: if test name is 'colonoscopy', everything is as normal. If test_name is 'new technology', the metrics for NT are used
    
    person -- a single row of the people array
    ppolyps -- the polyps array for a single person
    pcancers -- the cancers array for a single person
    ycosts -- {float} A cost in dollars. The costs accrued by this person will be added to ycosts
    kind -- {str} the reason for the colonoscopy. Can be 'surveillance', 'screening', 'symptom' or 'diagnostic'
    repeat -- {bool} whether or not this a repeat test due to a previous one having partial reach
    test_name -- {str} The name of the test to be used. Can be 'colonoscopy', 'new technology' if the new tech screener is a replacement/variant of colonoscopy
    prev_result -- {int} Designates the findings of the immediately preceding colonoscopy, in case this one is a repeat. Value meanings subject to change. Not used/assumed to be 0 if repeat==False
    
    Returns:
        person -- the updated person array
        ppolyps -- the updated ppolyps array
        pcancers -- the updated pcancers array
        ycosts -- the updated sum of costs
    '''

    endo_result = 0 # default if regular colo
    if kind == 'surveillance':
        endo_result = -10 # default if surv colo
    gets_polypectomy = False
    # NOTE: unlogged assumption: locCecum must be the highest valued location. If there is not full reach, detectable lesions will be in locations <= locCecum-1
    reach = globs.locCecum if (np.random.random() < .95) else globs.locCecum - 1

    if test_name == 'colonoscopy':
        SENS_CANCER = globs.SENS_CANCER_COLO
        SENS_SMALL = globs.SENS_SMALL_COLO
        SENS_MED = globs.SENS_MED_COLO
        SENS_LARGE = globs.SENS_LARGE_COLO
        SPECIFICITY = globs.SPEC_COLO
    else:
        SENS_CANCER = globs.SENS_CANCER_NT
        SENS_SMALL = globs.SENS_SMALL_NT
        SENS_MED = globs.SENS_MED_NT
        SENS_LARGE = globs.SENS_LARGE_NT
        SPECIFICITY = globs.SPEC_NT

    if kind == 'symptom':
        gets_polypectomy = True
    else:
        # get cancers and polyps that are active and within reach of this exam
        idx_detectable_cancers = (pcancers[:,globs.indLocationCancer] <= reach) & (pcancers[:,globs.indAgeCancerInit] <= person[globs.indAge]) & (pcancers[:, globs.indAgeCancerInit] > 0)
        idx_detectable_polyps = (ppolyps[:, globs.indLocationPolyp] <= reach) & (ppolyps[:,globs.indSizePolyp] > 0)

        if np.any(idx_detectable_cancers):
            # has cancer within reach
            num_active = int(np.sum(idx_detectable_cancers))
            idx_detected_cancer = (np.random.random(num_active) < SENS_CANCER)
            if np.any(idx_detected_cancer):
                # full diagnosis routines will be performed later in time step
                person[globs.indAgeDiagnosed] = person[globs.indAge]
                gets_polypectomy = True
                endo_result = -1 # indicates cancer

        if np.any(idx_detectable_polyps) and person[globs.indAgeDiagnosed] == 0: # has detectable polyps and wasn't just diagnosed with cancer
            # idx_detectable_polyps -> Bool(numPolypsAllowed, n)
            rands = np.random.random(int(np.sum(idx_detectable_polyps)))
            detected_polyps = np.zeros(len(rands))
            idx_small = ppolyps[idx_detectable_polyps, globs.indSizePolyp] < 6
            idx_med = (~idx_small) & (ppolyps[idx_detectable_polyps, globs.indSizePolyp] < 10)
            idx_large = (ppolyps[idx_detectable_polyps, globs.indSizePolyp] >= 10)
            detected_polyps[idx_small] = (rands[idx_small] < SENS_SMALL)
            detected_polyps[idx_med] = (rands[idx_med] < SENS_MED)
            detected_polyps[idx_large] = (rands[idx_large] < SENS_LARGE)
            idx_detected_polyps = (detected_polyps == 1) # convert to boolean array
            
            if np.any(idx_detected_polyps):
                gets_polypectomy = True
            # all detected polyps get resected
            to_resect = np.where(idx_detectable_polyps)[0] * idx_detected_polyps # numbered indices where there is a decetable polyp * whether each was detected -> numbered indices of where detected polyps are wrt to orginal polyps array
            # delete cancers that would have come from resected polyps
            for ind in ppolyps[to_resect, globs.indIsCancer]:
                pcancers[int(ind) - 1] = 0
            # delete polyps that are resected
            ppolyps[to_resect] = 0
            # determine time until next surveillance colo based on colnoscopy findings
            if (np.sum(detected_polyps) >= 3) | (np.sum(detected_polyps[idx_large]) >= 1): # at least one > 10mm or at least 3 of any size
                # next colo in 3 years
                endo_result = 3
            elif np.sum(detected_polyps) > 0:
                # next colo in 5 years
                endo_result = int(np.sum(detected_polyps))
                
        if (~ np.any(idx_detectable_cancers)) & (~np.any(idx_detectable_polyps)):
            # no active disease
            if np.random.random() > SPECIFICITY:
                # FP. Polypectomy still occurs (but the polyp is nonadenomatous)
                gets_polypectomy = True           
        
        if not repeat:
            # This was the first colonoscopy in the sequence
            if endo_result == -1:
                # Diagnosed with cancer. No more screening or surveillance
                person[globs.indAgeNextScreen] = 0
                person[globs.indAgeNextSurvCol] = 0
                person[globs.indAgeDiagnosed] = person[globs.indAge]
            elif endo_result == 0:
                # Nothing found. Screening in 10 years
                person[globs.indAgeNextScreen] = person[globs.indAge] + 10
                person[globs.indAgeNextSurvCol] = 0
            elif endo_result == -10:
                # if this was a surveillance colo, and nothing is found, get surveillance in 5 years still
                person[globs.indAgeNextScreen] = 0
                person[globs.indAgeNextSurvCol] = person[globs.indAge] + 5
            else:
                if endo_result == 3:
                    next_surv_colo = 3
                else:
                    next_surv_colo = 5
                # Surveillance is scheduled
                person[globs.indAgeNextSurvCol] = person[globs.indAge] + next_surv_colo
                person[globs.indAgeNextScreen] = 0
        else:
            # This colo happened right after a previous one that did not have full reach.
            # In this case, the findings of these colonoscopies need to be taken in conjunction
            if endo_result == -1 or prev_result == -1:
                # Cancer diagnosis
                person[globs.indAgeNextScreen] = 0
                person[globs.indAgeNextSurvCol] = 0
                person[globs.indAgeDiagnosed] = person[globs.indAge]
            elif endo_result + prev_result == 0:
                # No findings in diagnostic colo, return to screening in 10
                person[globs.indAgeNextScreen] = person[globs.indAge] + 10
                person[globs.indAgeNextSurvCol] = 0 
            elif endo_result == 3 or prev_result == 3:
                # Either of the colos found advanced polyps or 3+ nonadvanced, so surv colo in 3 years
                person[globs.indAgeNextSurvCol] = person[globs.indAge] + 3
                person[globs.indAgeNextScreen] = 0
            elif endo_result + prev_result >= 3:
                # The sum of detected polyps is at least 3, so next surv colo is in 3 years
                person[globs.indAgeNextSurvCol] = person[globs.indAge] + 3
                person[globs.indAgeNextScreen] = 0
            elif endo_result > 0 or prev_result > 0:
                # Some polyps were found in either, but not enough to warrant 3 year, so next surv colo is in 5 eyars
                person[globs.indAgeNextSurvCol] = person[globs.indAge] + 5
                person[globs.indAgeNextScreen] = 0
            elif endo_result + prev_result == -20:
                # No findings, but patient is still in surveillance
                person[globs.indAgeNextSurvCol] = person[globs.indAge] + 5
                person[globs.indAgeNextScreen] = 0
            else:
                print('surv scheduling error, endo, prev: ', endo_result, prev_result)
                         
    # Procedure costs
    if kind == 'screen':
        cost = globs.COST_SCREENING_COLO if test_name == 'colonoscopy' else globs.COST_NT
        person[globs.indCostScreen] += cost
        person[globs.indNumScreens] += 1
        ycosts += globs.COST_SCREENING_COLO
    elif kind == 'diagnostic':
        cost = globs.COST_DIAGNOSTIC_COLO if test_name == 'colonoscopy' else globs.COST_NT
        person[globs.indCostColonoscopy] += cost
        person[globs.indNumDiagColos] += 1
        ycosts += globs.COST_DIAGNOSTIC_COLO
    elif kind == 'surveillance':
        cost = globs.COST_SURVEILLANCE_COLO if test_name == 'colonoscopy' else globs.COST_NT
        person[globs.indCostColonoscopy] += cost
        person[globs.indNumSurveillanceColos] += 1
        ycosts += globs.COST_SURVEILLANCE_COLO
    elif kind == 'symptom':
        cost = globs.COST_SYMPTOM_COLO if test_name == 'colonoscopy' else globs.COST_NT
        person[globs.indCostColonoscopy] += cost
        person[globs.indNumSymptColos] += 1
        ycosts += globs.COST_SYMPTOM_COLO
    else:
        # NOTE: numba doesn't support these exceptions, so this won't even compile even this cell block is reached
        raise NotImplementedError('colo type is not implemented')
    
    # Complications and quality of life adjustments
    if gets_polypectomy:
        cost_complications = 0
        quality_adjustment = 0
        rands = np.random.random(5)
        if rands[0] < cardio_compl_prob(person[globs.indAge]):
            # cardio complication
            cost_complications += 10811.07
            quality_adjustment += globs.ADJ_COLO_COMPLICATION
        if rands[1] < minor_gi_compl_prob(person[globs.indAge]):
            # minor gi complication
            cost_complications += 8422.34
            quality_adjustment += globs.ADJ_COLO_COMPLICATION
        if rands[2] < major_gi_compl_prob(person[globs.indAge]):
            # serious gi complication
            cost_complications += 24039.86
            quality_adjustment += globs.ADJ_COLO_COMPLICATION
            if rands[3] < 0.089674 and rands[4]< 0.0519:
                # perforation and death from perforation
                person[globs.indAgeComplicationDeath] = person[globs.indAge]

        person[globs.indCostComplications] += cost_complications
        person[globs.indLYAdjustment] += quality_adjustment
        ycosts += cost_complications
    
    # quality adjustment for any colonoscopy
    person[globs.indLYAdjustment] += globs.ADJ_COLO
    
    # if this was not a repeat and full reach was not met, a repeat colo is performed with same adherence as screening ASSUMPTION THERE, THIS IS WHAT CRC-AIM DOES
    ADHERENCE = globs.ADHERENCE_COLO if test_name == 'colonoscopy' else globs.ADHERENCE_NT
    if (kind != 'symptom') & (not repeat) and (reach < globs.locCecum) and (np.random.random() < ADHERENCE):
        # NOTE: assume the kind of colonoscopy is diagnostic. Reconsider this assumption when the costs begin to differ between them
        # NOTE: if first colo was for symptoms, we DO NOT GET A SECOND
        #       this is based off of outcomes tables, which shows num_colos = num_cases
        #       if there were repeat symptom colos, num_colos would be 1/.96 * num_cases
        
        new_kind = kind
        person, ppolyps, pcancers, ycosts = get_colonoscopy(person, ppolyps, pcancers, ycosts, new_kind, True, test_name=test_name, prev_result=endo_result)

    if (person[globs.indAge] >= 85):
        if endo_result == 0 and prev_result == 0:
            # the last surv colo after age 85 was negative, so discontinue surveillance
            person[globs.indAgeNextSurvCol] = 0

    return person, ppolyps, pcancers, ycosts


@numba.njit
def get_screened_colo(people, polyps, cancers, ycosts, screen_name):
    '''Screening with colonoscopy (or an "invasive" screener)
    people -- the people array, subsetted to those getting an invasive screen
    polyps -- the polyps array, subsetted to those getting an invasive screen
    cancers -- the polyps array, subsetted to those getting an invasive screen
    ycosts -- {float} a value of cost. All costs across all people are added to this value
    screen_name -- {str} The name of the screener. Can be 'colonoscopy' or 'new technology'
    Returns:
        people -- the updated people array
        polyps -- the updated polyps array
        cancers -- the updated cancers array
        ycosts -- the updated sum of costs
    '''
    per_person_costs = np.zeros(len(people))
    for i in numba.prange(len(people)):
        year = int(people[i,globs.indAge])
        people[i], polyps[i], cancers[i], per_person_costs[i] = get_colonoscopy(people[i], polyps[i], cancers[i], 0, 'screen', False, test_name=screen_name)
    ycosts += np.sum(per_person_costs)
    return people, polyps, cancers, ycosts


@numba.njit
def get_screened_blood_stool(people, polyps, cancers, costs, screen_name):
    '''People get screened via a stool or blood-based screener (SCREEN_TYPE == 'noninvasive').
    We are assuming perfect adherence/compliance to followup colonoscopy
    Screen sensitivity is based on the most advanced lesion and results in a diagnostic colonoscopy if positive
    people -- the people array, subsetted to those getting screened
    polyps -- the polyps array, subsetted to those getting screened
    cancers -- the cancers array, subsetted to those getting screened
    costs -- {float} A value of costs. Costs accrued by all people in this routine will be added to costs
    screen_name -- {str} The name of the screener used
    returns:
        people -- the updated people array
        polyps -- the updated polyps array
        cancers -- the updated cancers array
        costs -- the updated total sum of costs
    '''
    
    if screen_name == 'cologuard':
        sensitivity_cancer, specificity = globs.SENS_CANCER_CG, globs.SPEC_CG
        sensitivity_small, sensitivity_med, sensitivity_large = globs.SENS_SMALL_CG, globs.SENS_MED_CG, globs.SENS_LARGE_CG
        screen_interval = globs.INTERVAL_CG
        screen_cost = globs.COST_CG
    elif screen_name == 'FIT':
        sensitivity_cancer, specificity = globs.SENS_CANCER_FIT, globs.SPEC_FIT
        sensitivity_small, sensitivity_med, sensitivity_large = globs.SENS_SMALL_FIT, globs.SENS_MED_FIT, globs.SENS_LARGE_FIT
        screen_interval = globs.INTERVAL_FIT
        screen_cost = globs.COST_FIT
    elif screen_name == 'new technology':
        sensitivity_cancer, specificity = globs.SENS_CANCER_NT, globs.SPEC_NT
        sensitivity_small, sensitivity_med, sensitivity_large = globs.SENS_SMALL_NT, globs.SENS_MED_NT, globs.SENS_LARGE_NT
        screen_interval = globs.INTERVAL_NT
        screen_cost = globs.COST_NT
    else:
        #NOTE: if running on server, this will probably just cause the thread to get loose and won't ever get the message (exceptions are not handled in NUMBA)
        print('This kind of screener is not implemented', screen_name)
        raise NotImplementedError("screener has not been implemented")
    

    for i, person in enumerate(people):
        age = int(person[globs.indAge])
        followup = False
        # determine most advanced lesion
        if np.any((cancers[i,:,globs.indAgeCancerInit] <= age) & (cancers[i,:,globs.indAgeCancerInit] > 0)):
            if np.random.random() < sensitivity_cancer:
                # TP. Get followup colonoscopy
                followup = True
            else:
                # FN. Assign time to next screen
                pass
        elif np.any(polyps[i,:,globs.indSizePolyp] > 0): #np.any((polyps[i,:,indAgeInitPolyp] <= age) & (polyps[i,:,indIsCancer])):
            max_size = np.max(polyps[i][(polyps[i,:,globs.indSizePolyp] > 0), globs.indSizePolyp]) #np.max(polyps[(polyps[i,:,indAgeInitPolyp] <= age), indSizePolyp])
            sensitivity = sensitivity_large if max_size >= 10 else sensitivity_med if max_size >= 6 else sensitivity_small
            if np.random.random() < sensitivity:
                # TP. Get followup colonoscopy
                followup = True
            else:
                # FN. Assign time to next screen
                pass
        else:
            # no disease
            if np.random.random() > specificity:
                # FP. Get followup colonoscopy
                followup = True
            else:
                # TN. Assign time to next screen
                pass
        
        if followup:
            # get diagnostic colonscopy
            person[:], polyps[i,:], cancers[i,:], costs = get_colonoscopy(person, polyps[i], cancers[i], costs, 'diagnostic', False)
        else:
            person[globs.indAgeNextScreen] = age + screen_interval # nothing was found. Screening resumes as normal

    # add cost of screen
    costs += screen_cost * len(people)
    people[:,globs.indCostScreen] += screen_cost
    people[:,globs.indNumScreens] += 1
    return people, polyps, cancers, costs


@numba.njit
def diagnose_cancer_symptomatic(people, cancers, costs, colo_type='colonoscopy'):
    '''Diagnosis of people who have symptomatic cancers this year
    
    people -- the people array, subsetted to those who have newly symptomatic cancer(s)
    cancers -- the cancers array, subsetted to those who have newly symptomatic cancer(s)
    costs -- {float} a value of cost. Costs accrued by all people are added to this value
    colo_type -- {str} Which test to use for the colonoscopy. Can be colonoscopy or new technology
    returns:
        people -- the updated people array
        cancers -- the updated cancers array
        costs -- the new sum of costs so far
    '''

    for i, person in enumerate(people):
        person_cancers = cancers[i]
        # determine cancer that is just symptomatic
        idx_active = (person_cancers[:,globs.indAgeCancerInit] <= int(person[globs.indAge])) & (person_cancers[:,globs.indAgeCancerInit] > 0)
        cancer_diagnosed = person_cancers[idx_active][np.argmin(person_cancers[idx_active,globs.indAgeAtSymptoms])]
        # get survival time of that cancer
        person[globs.indAgeCancerDeath] = int(person[globs.indAge] + cancer_diagnosed[globs.indSurvivalYearsSymptoms])
        # get stage of that cancer
        person[globs.indStageDiagnosed] = cancer_diagnosed[globs.indStageAtSymptoms]
        person[globs.indReasonDiagnosed] = globs.diagSymptoms
        # get colonoscopy/polypectomy
        temp_polyps = np.zeros_like(person_cancers) # no need to deal with actual polyps here. We assume all cancer is found and polyps aren't a concern
        people[i], _, _, costs = get_colonoscopy(person, temp_polyps, person_cancers, costs, 'symptom', False, test_name=colo_type)

    return people, cancers, costs


@numba.njit
def cardio_compl_prob(age):
    '''Gives the excess risk of cardiovascular complication due to colonoscopy w/ polypectomy
    Age is float or array of floats
    '''
    base_risk = 1/(np.exp(9.38297 - 0.07056 * age) + 1)
    pp_risk = 1/(np.exp(9.09053 - 0.07056 * age) + 1)
    return pp_risk - base_risk


@numba.njit
def minor_gi_compl_prob(age):
    '''Gives the excess risk of minor gastrointestinal complication due to colonoscopy w/ polypectomy
    Age is float or array of floats
    '''
    base_risk = 1/(np.exp(9.61197 - 0.05903 * age) + 1)
    pp_risk = 1/(np.exp(8.81404 - 0.05903 * age) + 1)
    return pp_risk - base_risk


@numba.njit
def major_gi_compl_prob(age):
    '''Gives the excess risk of serious gastrointestinal complication due to colonoscopy w/ polypectomy
    Age is float or array of floats
    '''
    base_risk = 1/(np.exp(10.78719 - 0.06105 * age) + 1)
    pp_risk = 1/(np.exp(9.27953 - 0.06105 * age) + 1)
    return pp_risk - base_risk


#NOTE: try all possible solutions
@numba.njit
def diagnose_cancer_colonoscopy(people, cancers, cancer_stage_pars):
    '''Diagnosis procedures, following diagnoses from colonoscopies (surveillance, screen, diagnostic colonoscopy)
    people -- the people array, subsetted to those who had cancer detected by a colonoscopy
    cancers -- the cancers array, subsetted to those who had cancer detected by a colonoscopy
    cancer_stage_pars -- the parameters used to assign stage of cancer based on size
    Returns:
        people -- the updated people array
        cancers -- the updated cancers array
    '''

    for i, person in enumerate(people):
        # we don't keep track of which cancer was diagnosed/caught, as we assume all active cancers will be found. 
        # So get size/stage/survival for all active cancers
        idx_active = (cancers[i][:,globs.indAgeCancerInit] <= person[globs.indAge]) & (cancers[i][:,globs.indAgeCancerInit] > 0)
        for j, cancer in enumerate(cancers[i]):
            if ((cancer[globs.indAgeCancerInit] > person[globs.indAge]) | (cancer[globs.indAgeCancerInit] == 0)):
                continue
            # Compute size at diagnosis
            b = cancer[globs.indCancerGrowthRate]
            a = .5 # min cancer size
            t = person[globs.indAge] - cancer[globs.indAgeCancerInit] # NOTE: this will always be an integer number of years. Is that okay??
            c_size =  a * b**t
            # Determine stage
            stage = get_cancer_stage_given_size(c_size, cancer_stage_pars)
            # Survival time from this cancer
            survival_years = get_death_from_cancer_age(person[globs.indAge], person[globs.indGender], person[globs.indLocationCancer], stage)
            cancers[i,j,globs.indSizeAtDetect] = c_size
            cancers[i,j,globs.indStageAtDetect] = stage
            cancers[i,j,globs.indSurvivalYearsDetect] = survival_years
        
        ## Possible death ages
        #death_ages = person[indAge] + cancers[i][idx_active,indSurvivalYearsDetect]
        #death_age = np.min(death_ages)
        ## Possible symptom ages
        #symptom_ages = cancers[i][idx_active,indSojournTCancer] + cancers[i][idx_active,indAgeCancerInit]
        #symptom_age = np.min(symptom_ages)
        
        # NOTE: for now, assume that the death_age is min(symptom_ages_i + survival_years'_i)
        symptom_ages = cancers[i][idx_active,globs.indSojournTCancer] + cancers[i][idx_active,globs.indAgeCancerInit]
        survival_years = cancers[i][idx_active,globs.indSurvivalYearsDetect]
        
        person[globs.indAgeCancerDeath] = int(np.min(symptom_ages + survival_years))
        # NOTE: Let's try something else?

        # Cancer death age given possible death and symptom ages
        #person[indAgeCancerDeath] = int(max((death_age, symptom_age))) # if cancer death is before end of dwell/sojourn time, wait until symptomatic to die from cancer
        most_advanced_cancer = cancers[i][np.argmax(cancers[i][idx_active,globs.indStageAtDetect])]
        person[globs.indStageDiagnosed] = most_advanced_cancer[globs.indStageAtDetect]
        person[globs.indLocDiagnosed] = most_advanced_cancer[globs.indLocationCancer]
        #person[globs.indStageDiagnosed] = np.max(cancers[i][idx_active,globs.indStageAtDetect]) # stage is the max of active cancers, not the one that kills first
        #person[globs.indLocDiagnosed] = cancers[i][np.argmax(cancers[i][idx_active,globs.indStageAtDetect]),globs.indLocDiagnosed]
        person[globs.indReasonDiagnosed] = globs.diagIntervention

    return people, cancers


def validate(num_people=int(1e6), cores=5, client=None):
    '''Runs a full simulation and prints key outcomes, along with targets, in order to revalidate model in the event of any changes
    NOTE: this validates assuming targets are screening 50-80, full adherence, period life table, 1975 byear
    These must be the default in globs, as they are not explicitly set here
    num_people -- {int} The number of people to run, if running natural history
    cores -- {int} The number of asynchronous processes to spawn
    client -- BigQuery client. If None, who simulation including natural history is run. If a Client, existing natural history is used
        NOTE: currently will not work if client is not None!!!!!!
    '''
    if client is None:
        all_res = run(int(1e1))
        print('done compiling')
        all_res = run(int(num_people), n_jobs=cores)
        print('done running')
        results_none, results_colo, results_cg, results_fit, results_new = all_res[3], all_res[5], all_res[7], all_res[9], all_res[11]
    else:
        results_none, _, results_colo, _, results_cg, _, results_fit, _ = get_SOC_stored_results(client)
    
    all_res = [results_none, results_colo, results_cg, results_fit, results_new]
    for i in range(len(all_res)):
        df = all_res[i]
        try:
            all_res[i] = df.loc[((df.Death_Age > 40) & ~(df.Diag_Age <= 40))]
        except AttributeError as ae:
            print(i)
            print(df.columns)
            print(all_res[i].columns)
            print(df)
            raise ae
    
    none_eligible = results_none.loc[((results_none.Death_Age > 40) & ~(results_none.Diag_Age <= 40))]
    colo_eligible = results_colo.loc[((results_colo.Death_Age > 40) & ~(results_colo.Diag_Age <= 40))]
    cg_eligible = results_cg.loc[((results_cg.Death_Age > 40) & ~(results_cg.Diag_Age <= 40))]
    fit_eligible = results_fit.loc[((results_fit.Death_Age > 40) & ~(results_fit.Diag_Age <= 40))]
    new_eligible = results_new.loc[((results_new.Death_Age > 40) & ~(results_new.Diag_Age <= 40))]
    screeners = ['none', 'colo', 'cg', 'fit', 'nt']
    screener_results = [none_eligible, colo_eligible, cg_eligible, fit_eligible, new_eligible]

    stage_rates = [.16,.38,.26,.19]
    for stage in [1,2,3,4]:
        print('Stage {}: me {:.3f}, crcaim {:.3f}'.format(stage, np.mean(none_eligible.loc[none_eligible.Diag_Stage > 0].Diag_Stage == stage), stage_rates[stage-1]))

    # for those cancer free at 65, lets look at 10 year and lifetime cumulative incidence of cancer diagnosis
    cancer_free_65 = none_eligible.loc[none_eligible.Death_Age > 65 & ~(none_eligible.Diag_Age <= 65)]
    print('these were generated in report with cohort tables, though')
    print('10 year incidence, cancer free at 65: me {:.3f}, crcaim {}'.format(np.mean(cancer_free_65.Diag_Age < 75), .023))
    print('lifetim incidence, cancer free at 65: me {:.3f}, crcaim {}'.format(np.mean(cancer_free_65.Diag_Age < 120), .064))
    validation = {
        # 
        'cases':{'none_true':71.1, 'colo_true':9.2, 'cg_true':23.2, 'fit_true':20.2},
        'colos':{'none_true':71, 'colo_true':4451, 'cg_true':1969, 'fit_true':2052},
        'sympt_colos':{'none_true':71, 'colo_true':5, 'cg_true':11, 'fit_true':9},
        'diag_colos':{'none_true':0, 'colo_true':0, 'cg_true':874, 'fit_true':874},
        'surv_colos':{'none_true':0, 'colo_true':1650, 'cg_true':1085, 'fit_true':1169},
        'screens':{'none_true':0, 'colo_true':2796, 'cg_true':6517, 'fit_true':17225},
        'crc_deaths':{'none_true':31.7, 'colo_true':3.2, 'cg_true':7.2, 'fit_true':5.8},
        'lyg':{'none_true':0, 'colo_true':297.8, 'cg_true':258.7, 'fit_true':273}
    }
    for name, df in zip(screeners, screener_results):
        validation['cases'][name+'_me'] = np.mean(df.Diag_Age > 0) * 1000
        validation['colos'][name+'_me'] = np.mean(df.NumColos) * 1000
        validation['sympt_colos'][name+'_me'] = np.mean(df.NumSymptColos) * 1000
        validation['diag_colos'][name+'_me'] = np.mean(df.NumDiagColos) * 1000
        validation['surv_colos'][name+'_me'] = np.mean(df.NumSurvColos) * 1000
        validation['screens'][name+'_me'] = np.mean(df.NumScreens) * 1000
        validation['crc_deaths'][name+'_me'] = np.mean(df.Death_Cause != 'other') * 1000
        validation['lyg'][name+'_me'] = np.mean(df.Death_Age - none_eligible.Death_Age) * 1000
    for name in screeners:
        if name == 'nt':
            continue
        for metric in validation.keys():
            if name == 'none' and metric in ['lyg', 'screens', 'diag_colos', 'surv_colos']:
                validation[metric][name+'_err'] = 0
                continue
            if name == 'colo' and metric == 'diag_colos':
                validation[metric][name+'_err'] = 0
                continue
            validation[metric][name+'_err'] = (validation[metric][name+'_me'] - validation[metric][name+'_true']) / validation[metric][name+'_true']

    validation = pd.DataFrame.from_dict(validation)
    pd.options.display.float_format = "{:,.2f}".format
    print(validation)
    
    all_res = [results_none, results_colo, results_cg, results_fit]
    for i in range(len(all_res)):
        df = all_res[i]
        all_res[i] = df.loc[((df.Death_Age > 40) & ~(df.Diag_Age <= 40))]
    
    # num screens
    results = {}
    screeners = ['none', 'colonoscopy', 'cologuard', 'fit']
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.NumScreens)*1000
    results['num_screens'] = temp

    # diagnosed cancers
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.Diag_Age > 0) * 1000
    results['crc_cases'] = temp

    # surveillance colos
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.NumSurvColos) * 1000
    results['surv_colos'] = temp

    # diagnostic colos
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.NumDiagColos) * 1000
    results['diag_colos'] = temp

    # symptom colos
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.NumSymptColos) * 1000
    results['sympt_colos'] = temp
    
    # CRC deaths
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.Death_Cause == 'cancer') * 1000
    results['crc_deaths'] = temp

    # LYG
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = (np.mean(df.Death_Age) - np.mean(all_res[0].Death_Age)) * 1000
    results['LYG'] = temp
    
    # Cost per member
    temp = {}
    for name, df in zip(screeners, all_res):
        temp[name] = np.mean(df.Costs)
    results['perMemberCost'] = temp
    
    results = pd.DataFrame.from_dict(results)
    print(results)
    
    return


if __name__ == "__main__":
    #validate(num_people=4e5, cores=8)
    
    from google.cloud import bigquery
    from google.oauth2 import service_account
    from google.cloud import storage
    import time
    
    service_account_path = "./creds/svc-int-dev-crc.json"

    credentials = service_account.Credentials.from_service_account_file(
        service_account_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform'],)

    import time
    #import pydata_google_auth as pgauth
    
    client = bigquery.Client(credentials=credentials,project=credentials.project_id,)
    
        
    user_stack = {
      "sensitivity":{"CRC":0.95,"Polyps<6mm":0.75,"6mm<Polyps<10mm":0.85,"10mm<Polyps":0.95}, 
      "specificity":0.86, 
      "cost":1337.46,
      "adherence":1.0,
      "interval":10,
      "kind":'invasive'}

    window = [50, 80]
    t0 = time.time()
    results = run_app(client, user_stack, window, n_jobs=10)
    t1 = time.time()
    print(results)
    print('Total runtime: {:.3f} minues'.format((t1-t0)/60))
        
