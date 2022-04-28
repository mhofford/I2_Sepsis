import pandas as pd
import numpy as np
import time
import datetime
import i2bmi
import matplotlib.pyplot as plt
from scipy import stats


def merge_between(left_df, right_df,time_col, occurs_after, occurs_before, on, splits=100, after_margin=pd.Timedelta('0h'),before_margin = pd.Timedelta('0h')):
    '''Merges All results in a time period by merging all then applying the time critera, splits into multiple dataframes to avoid memory issues
    Improvements to pandas to eliminate this work around are currently ongoing: https://github.com/pandas-dev/pandas/issues/8962'''


    result = {}
    dfs = np.array_split(left_df, splits)
    for count, df in enumerate(dfs):
        df[on] = df[on].astype(str)
        right_df[on] = right_df[on].astype(str)

        df2 = pd.merge(df, right_df, on=on, how='left')

        df2[time_col] = pd.to_datetime(df2[time_col],utc=True)
        df2[occurs_before] = pd.to_datetime(df2[occurs_before], utc = True)
        df2[occurs_after] = pd.to_datetime(df2[occurs_after], utc = True)


        crit_1 = df2[time_col] >= (df2[occurs_after] - after_margin)
        crit_2 = df2[time_col] <= (df2[occurs_before] + before_margin)
        result[f'split_{count}'] = df2[crit_1 & crit_2]
    return pd.concat(result, ignore_index=True)


def perdelta(start, end, delta):
    '''Creates times to output based on the selected delta start and end in the make output function'''
    curr = start
    while curr < end + delta:
        yield curr
        curr += delta


def make_output(adt=None, calc_interval_mins=None):
    '''Makes an output dataframe based on the admissions discharge and transfer dataframe as well as the user speficifed interval'''

    # Find Start of Encounters
    adt = adt.sort_values(by=['patient_id', 'encounter_id', 'location_start'], ascending=True)
    first = pd.DataFrame(adt.groupby(['patient_id', 'encounter_id'])['location_start'].first()).reset_index()
    first.rename(columns={'location_start': 'start'}, inplace=True)
    # Find End of Encounters
    adt = adt.sort_values(by=['patient_id', 'encounter_id', 'location_end'], ascending=True)
    last = pd.DataFrame(adt.groupby(['patient_id', 'encounter_id'])['location_end'].last()).reset_index()
    last.rename(columns={'location_end': 'end'}, inplace=True)

    temp = pd.merge(first, last, on=['patient_id', 'encounter_id'])

    # Add Rows at user specified interval
    temp['score_time'] = temp.apply(
        lambda row: list(perdelta(row.start, row.end, datetime.timedelta(minutes=calc_interval_mins))), axis=1)
    output = temp[['patient_id', 'encounter_id', 'score_time']].explode('score_time').reset_index(drop=True)

    # Add a row at the time of transfer from one location to another
    starts = adt[['patient_id', 'encounter_id', 'location_start']].rename(columns={'location_start': 'score_time'})
    ends = adt[['patient_id', 'encounter_id', 'location_end']].rename(columns={'location_end': 'score_time'})

    output = pd.concat([output,starts, ends])
    output = output.dropna().drop_duplicates().sort_values(by = ['patient_id', 'encounter_id','score_time']).reset_index(drop=True)
    output = output.dropna()

    return output


def convert_pressors(vasodf=None,
                     pressor_map={'dobutamine': 'dobu', 'dopamine': 'dopa', 'milrinone': 'milrinone',
                                  'epinephrine': 'epi', 'norepinephrine': 'norepi', 'phenylephrine': 'phenyl',
                                  'vasopressin': 'vasopressin'},
                     unit_key={'dobu': 'mcg/kg/min', 'dopa': 'mcg/kg/min', 'milrinone': 'mcg/kg/min',
                               'ang': 'ng/kg/min', 'epi': 'mcg/kg/min', 'norepi': 'mcg/kg/min', 'phenyl': 'mcg/kg/min',
                               'vasopressin': 'Units/min'},
                     NE_eqs={'norepi': 1, 'epi': 1, 'dopa': 15 / 0.1, 'vasopressin': 0.04 / 0.1, 'phenyl': 1.0 / 0.1,
                             'ang': 10 / 0.1}):
    '''Converts pressors to norepi equivilants based on user specified conversion factors, also removes incorrect units if present
    default conversions based on Lambden S, Laterre PF, Levy MM, Francois B. The SOFA score-development, utility and challenges of accurate assessment in clinical trials. Crit Care.
     2019 Nov 27;23(1):374. doi: 10.1186/s13054-019-2663-7. PMID: 31775846; PMCID: PMC6880479.'''

    # Check if pressors are correctly mapped
    if len(set(vasodf['pressor_name']).intersection(set(pressor_map.keys()))) > 2:
        print('Converting Pressor Names')
        vasodf['pressor_name'] = vasodf['pressor_name'].map(pressor_map)
        #display(pd.DataFrame(vasodf['pressor_name'].value_counts()))
    else:
        print('No Pressor Name Conversion Needed')

    # if vasopressin is in units per hour convert to units per min
    vasodf.loc[(vasodf['pressor_name'] == 'vasopressin') & (vasodf['vaso_units'] == 'units/hour'), 'vaso_dose'] = \
    vasodf.loc[(vasodf['pressor_name'] == 'vasopressin') & (vasodf['vaso_units'] == 'units/hour'), 'vaso_dose'] / 60
    vasodf.loc[
        (vasodf['pressor_name'] == 'vasopressin') & (vasodf['vaso_units'] == 'units/hour'), 'vaso_units'] = 'units/min'

    # drop lines that do not have the included units
    before = vasodf
    vasodf = vasodf[vasodf['vaso_units'] == vasodf['pressor_name'].map(unit_key)].reset_index(drop=True)
    after = vasodf
    dropped_units = set(before['vaso_units']) - set(after['vaso_units'])
    if len(before) - len(after) > 0:
        print('{} entries dropped due to incorrect units \n Units dropped were {}'.format(len(before) - len(after), dropped_units))
    else:
        print('No entries dropped due to incorrect units')

    # Specify Equivilents
    ianotropes = {'dopa': 1, 'dobu': 1, 'milrinone': 1}
    vasodf['vaso_dose'] = pd.to_numeric(vasodf['vaso_dose'], errors='coerce')


    # Deal with dopamine dose <= 5
    mask = (vasodf['pressor_name'] == 'dopa') & (vasodf['vaso_dose'] <= 5)
    vasodf.loc[mask, 'ianotrope'] = 1
    vasodf.loc[mask, 'vaso_dose'] = 0

    #convert all other pressors to NE eqs
    vasodf['ne_eq'] = (vasodf['vaso_dose'] / vasodf['pressor_name'].map(NE_eqs)).fillna(0)
    vasodf['ianotrope'] = vasodf['pressor_name'].map(ianotropes).fillna(0)

    #check for issues
    if (any(vasodf['ne_eq'].isna())) | (any(vasodf['ianotrope'].isna())):
        raise Exception('Some medications not found')

    # Trim columns
    pressors = vasodf[['patient_id', 'pressor_name', 'vaso_start', 'vaso_end', 'ianotrope', 'ne_eq']]

    return pressors


def Flow_to_FiO2(lvdf=None, max_flow_convert=6):
    '''Converts O2 Flow to FIO2 Based on:
    Yu SCBetthauser KD, Gupta A, Lyons PG, Lai AM, Kollef MH, Payne PRO, Michelson AP.
    Comparison of Sepsis Definitions as Automated Criteria.
    Crit Care Med. 2021 Apr 1;49(4):e433-e443.
    doi: 10.1097/CCM.0000000000004875. PMID: 33591014.
    '''
    print('Estimating FiO2 from Flow Rate for Flow <= {} LPM'.format(max_flow_convert))

    mylvdf = lvdf[lvdf['label'] == 'O2_Flow'].reset_index(drop=True)
    mylvdf['value'] = mylvdf['value'].astype(float)
    mylvdf = mylvdf[mylvdf['value'] <= max_flow_convert].reset_index(drop=True) 
    mylvdf.loc[:, 'value'] = ((3.5 * mylvdf['value']) + 21) / 100
    mylvdf.loc[:, 'label'] = 'FiO2'
    lvdf = lvdf.append(mylvdf, ignore_index=True)
    return lvdf


def Calc_PF(lvdf=None, include_SF=False, tolerance=1):
    '''Calculates PO2/FiO2 Ratio with option to also caculate SpO2/FiO2 rato default tolerance to compute is 1 hours
    for more information on Sp/FiO2 ratio:
    Pandharipande PP, Shintani AK, Hagerman HE, St Jacques PJ, Rice TW, Sanders NW, Ware LB, Bernard GR, Ely EW. Derivation and validation of Spo2/Fio2 ratio to impute for Pao2/Fio2 ratio in the respiratory component of the Sequential Organ Failure Assessment score. Crit Care Med. 2009 Apr;37(4):1317-21. doi: 10.1097/CCM.0b013e31819cefa9. PMID: 19242333; PMCID: PMC3776410.
    '''

    data = {}
    numerator = ['SpO2', 'pO2']
    denominator = 'FiO2'

    print('Calculating P/F Ratio')
    if include_SF == False:
        numerator.remove('SpO2')
        print('Not Considering SpO2/F Ratio')
    else:
        print('Calculating SpO2/F Ratio as well')

    mylvdf = lvdf[(lvdf['label'].isin(numerator)) | (lvdf['label'] == denominator)].reset_index(drop=True)
    mylvdf.loc[:, 'value'] = mylvdf['value'].astype(float)

    for num in numerator:
        top = mylvdf[mylvdf['label'] == num].rename(columns={'label': '1 label', 'value': num})
        bot = mylvdf[mylvdf['label'] == denominator].rename(columns={'label': '2 label', 'value': denominator})

        # convert to fraction if expressed as percent O2
        bot[denominator] = bot[denominator].map(lambda a: a / 100 if a > 1 else a)
        bot = bot.groupby(['patient_id', 'time_measured', '2 label'])[denominator].max().reset_index()

        top.sort_values(by='time_measured', inplace=True)
        bot.sort_values(by='time_measured', inplace=True)

        merge = pd.merge_asof(top, bot, on='time_measured', by='patient_id', allow_exact_matches=True,
                              direction='backward', tolerance=pd.Timedelta(f'{tolerance}h'))

        if num == 'pO2':
            merge.loc[:, 'label'] = 'PFRatio'

        elif num == 'SpO2':
            merge.loc[:, 'label'] = 'SFRatio'

        merge.loc[:, 'value'] = merge[num] // merge[denominator]
        merge['value'].replace([np.inf, -np.inf], np.nan, inplace=True)

        data[num] = merge[['patient_id', 'value', 'time_measured', 'label']]

    mylvdf = pd.concat(data, ignore_index=True).reset_index(drop=True)
    lvdf = lvdf.append(mylvdf, ignore_index=True)
    return lvdf

def search_lvdf(lvdf= None, output=None, scorename = None, elementdict = None, calc_interval_mins = 60 ,find_high_hrs=24, LOCF_hours = None, debug= False):
    '''Finds the measure of interest in labs and vitals dataframe and calculated the specified score'''
    
    output = output.drop('output_id', axis=1, errors='ignore') # drop output_id if present
    startingshape = output.shape[0]

    lvdf['time_measured'] = pd.to_datetime(lvdf['time_measured'],utc=True)
    output['score_time'] = pd.to_datetime(output['score_time'],utc=True)

    #make output ID
    output.reset_index(inplace=True)
    output.rename(columns={'index':'output_id'},inplace=True)
    
    
    print('\nSearching lvdf for {} data \n ======================================'.format(scorename))
    if set(elementdict.keys()).issubset(set(lvdf['label'])):
        print('All Vars are present in the dataset (Individual patients may still be missing data)',end = '\n\n')
    else:
        missing = list(set(elementdict.keys()) - set(lvdf['label']))
        print('{} missing from the dataset'. format(missing),end='\n\n')
        for elem in missing: 
            elementdict.pop(elem)
            elementscorename='{}_{}_Score'.format(scorename,elem)
            output[elementscorename] = np.NaN
            

    mylvdf = lvdf.loc[lvdf['label'].isin(list(elementdict.keys()))].copy()
    

    for element in elementdict:
        output_trim = output.loc[:,['output_id','patient_id','score_time']]
        output_trim['patient_id'] = output_trim['patient_id'].astype(str)
    
        print('Processing {} for {} score...'.format(element,scorename),end='')
        curtime=time.time()

        #Slice lvdf for each element,
        element_df = mylvdf.loc[mylvdf['label']==element,['patient_id','time_measured','value','label']].copy()
        element_df['value'] = pd.to_numeric(element_df['value'], errors='coerce')
        element_df['patient_id'] = element_df['patient_id'].astype(str)

        output_trim = output_trim.sort_values(by = ['score_time'])
        element_df = element_df.sort_values(by = ['time_measured'])

        #score all elements
        elementscorename = '{}_{}_Score'.format(scorename,element)
        element_df[elementscorename]  = elementdict[element]['default']

        for fx in elementdict[element]['fx']:
            element_df.loc[element_df.eval(fx['expr']),elementscorename] = fx['score']

        # make sure element exists in lv
        assert element_df.shape[0]>0, '{} was not found in lv'.format(element)

        # (MAXRECENT) filter by short interval -> take max score per element per ID-outputtime
        short_int_df  = merge_between(left_df=output_trim, right_df=element_df,time_col='score_time', occurs_after = 'time_measured', occurs_before = 'time_measured', on='patient_id', splits=100,
                      after_margin=pd.Timedelta('0h'), before_margin=pd.Timedelta(f'{find_high_hrs}h'))
        short_int_df['patient_id'] = short_int_df['patient_id'].astype(str)

        #Keep highest scoring element in the short timeframe
        short_int_df = short_int_df.sort_values(by=['patient_id','score_time',elementscorename],ascending=False).groupby(['patient_id','score_time']).first().reset_index()

        output_trim['output_id'] = output_trim['output_id'].astype(int)
        short_int_df['output_id']= short_int_df['output_id'].astype(int)
        short_int_df = pd.merge(output_trim,short_int_df[['output_id','value','time_measured',elementscorename]],on=['output_id'], how ='left').sort_values(by='output_id')

        if LOCF_hours is None:
            print('No LOCF Considered')
            final = short_int_df.sort_values(by ='output_id')

        else:
            # merge as of to do LOCF
            output_trim = output_trim.sort_values(by='score_time')
            element_df = element_df.sort_values(by='time_measured')

            LOCF_long = pd.merge_asof(output_trim, element_df, left_on = 'score_time', right_on = 'time_measured', by = 'patient_id', tolerance= datetime.timedelta(hours=LOCF_hours),direction='backward').sort_values(by='output_id')

            # Make sure dataframes are in the same order
            short_int_df.sort_values(by = 'output_id',inplace = True)
            LOCF_long.sort_values(by = 'output_id',inplace = True)
        
            if len(short_int_df) != len(LOCF_long):
                raise Exception('Short and Long Timeframe dataframes are not the same length')

            # Add LOCF if Value is missing
            final = short_int_df.combine_first(LOCF_long)

        if debug:
            output = output.merge(final.loc[:,['output_id','patient_id', 'score_time','value', 'time_measured', elementscorename]], on= ['output_id','patient_id','score_time'], how = 'left')
            output.rename(columns = {'value':element,'time_measured': element + '_time'},inplace = True)
            output[elementscorename] = output[elementscorename].fillna(0)
            print('Percent Missing: {:.2f} '.format(output[element].isna().mean()*100),end='')
            
        else:
            output = output.merge(final.loc[:,['output_id','patient_id','score_time', elementscorename]], on= ['output_id','patient_id','score_time'], how = 'left')
            output[elementscorename] = output[elementscorename].fillna(0)
        print('---Complete ({:.1f}s)'.format(time.time()-curtime),end='\n\n')

    output = output.sort_values(by = ['patient_id','score_time'])

    #Check that shape did not change
    assert output.shape[0]==startingshape, 'Something is wrong, number of rows changed'
    return output


def find_vent(mvdf=None, output=None,  debug=False, mech_vent_def=None):
    '''finds when patients are on mechanical ventilation'''
    assert mech_vent_def in ['VENT','VENT/NIPPV']

    if mech_vent_def == 'VENT':
        print('Enforcing mechanical ventialation as VENT only')
        mvdf_trim = mvdf[mvdf['vent_type']=='VENT'].reset_index(drop=True).copy()
    else:
        print('Not enforcing mechanical ventialation as VENT only')
        mvdf_trim = mvdf.copy()

    output['score_time'] = pd.to_datetime(output['score_time'], utc=True)
    mvdf_trim['vent_start'] = pd.to_datetime(mvdf_trim['vent_start'], utc=True)
    mvdf_trim['vent_end'] = pd.to_datetime(mvdf_trim['vent_end'], utc=True)


    # consider patient vented if on vent in last 24 hours:
    mvdf_trim['vent_end'] = mvdf_trim['vent_end'] + pd.Timedelta('24 hours')

    mvdf_trim.sort_values(by='vent_start', inplace=True)
    output.sort_values(by='score_time', inplace=True)

    output = pd.merge_asof(output, mvdf_trim, left_on='score_time', right_on='vent_start', by='patient_id',
                           allow_exact_matches=True, direction='backward').reset_index(drop=True)
    output.loc[output['score_time'].between(output['vent_start'], output['vent_end']), 'vent'] = 1
    output['vent'].fillna(0, inplace=True)

    if not debug:
        output.drop(columns=['vent_start', 'vent_end'])

    return output

def find_pressor(vasodf=None, output=None, debug=False, find_high_hrs=24):
    '''Finds max total pressor dose in norepi-eqs in the last specified time period,'''

    output['patient_id'] = output['patient_id'].astype(int).astype(str)
    output['encounter_id'] = output['encounter_id'].astype(int).astype(str)

    output['score_time'] = pd.to_datetime(output['score_time'], utc=True)
    vasodf['vaso_start'] = pd.to_datetime(vasodf['vaso_start'], utc=True)
    vasodf['vaso_end'] = pd.to_datetime(vasodf['vaso_end'], utc=True)

    pressors = convert_pressors(vasodf=vasodf)
    pressors.dropna(subset=['vaso_start', 'vaso_end'], inplace=True)
    pressors.sort_values(by='vaso_start', inplace=True)
    output.sort_values(by='score_time', inplace=True)

    output_trim = output.loc[:, ['patient_id', 'encounter_id', 'score_time']]


    find_current = merge_between(left_df=output_trim, right_df = pressors, time_col='score_time', occurs_after='vaso_start', occurs_before='vaso_end', on='patient_id', splits=100,
                  after_margin=pd.Timedelta(f'0h'), before_margin=pd.Timedelta(f'{find_high_hrs}h'))

    find_current['patient_id'] = find_current['patient_id'].astype(int).astype(str)
    find_current['encounter_id'] = find_current['encounter_id'].astype(int).astype(str)
    find_current['score_time'] = pd.to_datetime(find_current['score_time'], utc=True)

    iano_current = find_current.copy()
    #Take Most recent dose of each pressor
    find_current['time_from_start'] =  (find_current['score_time'] - find_current['vaso_start']).apply(lambda z: z.total_seconds()/3600)
    find_current = find_current.sort_values(by='time_from_start',ascending=True)
    assert find_current['time_from_start'].min() >=0
    find_current = find_current.groupby(['patient_id', 'encounter_id', 'score_time', 'pressor_name']).first().reset_index()
    NE_equiv = find_current.groupby(['patient_id', 'encounter_id', 'score_time'])['ne_eq'].sum().reset_index()

    #find_current.rename(columns={'score_time':'pressor_time'},inplace=True)
    #find_current = find_current[['patient_id','pressor_time','ne_eq']]

    #NE_equiv = merge_between(left_df=output_trim, right_df = find_current, time_col='score_time', occurs_after='pressor_time', occurs_before='pressor_time', on='patient_id', splits=100,
                  #after_margin=pd.Timedelta(f'0h'), before_margin=pd.Timedelta(f'{find_high_hrs}h'))

    #NE_equiv = NE_equiv.sort_values(by = 'ne_eq',ascending=False)
    #NE_equiv = NE_equiv.groupby(['patient_id', 'encounter_id', 'score_time'])['ne_eq'].first().reset_index()


    # Find iano
    iano = iano_current.groupby(['patient_id', 'encounter_id', 'score_time'])['ianotrope'].max().reset_index()
    final = pd.merge(iano, NE_equiv, on=['patient_id', 'encounter_id', 'score_time'], how='outer')

    output['patient_id'] = output['patient_id'].astype(int).astype(str)
    output['encounter_id'] = output['encounter_id'].astype(int).astype(str)
    output = pd.merge(output, final, on=['patient_id', 'encounter_id', 'score_time'], how='left')

    if debug:
        output.rename(columns={'time_measured': 'vaso_time'}, inplace=True)
    else:
        output.drop(columns='time_measured')
    output_trim.fillna(0)

    output = output.merge(output_trim, on=['patient_id', 'encounter_id', 'score_time'], how='left')
    return output


def find_uo24(uodf=None, output=None, debug=False, find_low_hrs=24):

    uodf = uodf.dropna().reset_index(drop=True)
    output['patient_id'] = output['patient_id'].astype(int).astype(str)
    output['encounter_id'] = output['encounter_id'].astype(int).astype(str)
    uodf['patient_id'] = uodf['patient_id'].astype(int).astype(str)

    output['score_time'] = pd.to_datetime(output['score_time'], utc=True)
    uodf['uo_time'] = pd.to_datetime(uodf['uo_time'], utc=True)

    uodf.sort_values(by='uo_time', inplace=True)
    output.sort_values(by='score_time', inplace=True)

    output_trim = output.loc[:, ['patient_id', 'encounter_id', 'score_time']]


    find_current = merge_between(left_df = output_trim, right_df=uodf, time_col = 'score_time', occurs_after= 'uo_time', occurs_before='uo_time', on='patient_id', splits=100,
                                 after_margin=pd.Timedelta('0h'), before_margin=pd.Timedelta(f'{find_low_hrs}h') )

    find_current['patient_id'] = find_current['patient_id'].astype(int).astype(str)
    find_current['encounter_id'] = find_current['encounter_id'].astype(int).astype(str)
    find_current['score_time'] = pd.to_datetime(find_current['score_time'], utc=True)
    find_current['uo_time'] = pd.to_datetime(find_current['uo_time'], utc=True)

    find_current.dropna(inplace=True)

    find_current = find_current.groupby(['patient_id', 'encounter_id', 'score_time']).agg(
        {'uo_24hr': 'min'}).reset_index()

    return find_current


def score_qSOFA(lvdf=None, adt=None, calc_interval_mins=None, LOCF_hours=None, debug=False, gcs_cutoff=15, cutoff=2):
    '''Calculates qSOFA score for cohort'''

    print('Warning: qSOFA: Using GCS < {} for AMS'.format(gcs_cutoff))

    scorename = 'qSOFA'

    elementdict = {
        'SBP': {'default': 0, 'fx': [
            {'expr': '({}<=100)'.format('value'), 'score': 1},
        ]},
        'RR': {'default': 0, 'fx': [
            {'expr': '({}>=22)'.format('value'), 'score': 1},
        ]},
        'GCS': {'default': 0, 'fx': [
            {'expr': '({}<{})'.format('value', gcs_cutoff), 'score': 1},
        ]},
    }

    output = make_output(adt=adt, calc_interval_mins=calc_interval_mins)

    output = search_lvdf(lvdf=lvdf, output=output, scorename=scorename, calc_interval_mins=calc_interval_mins,
                         LOCF_hours=LOCF_hours, elementdict=elementdict, debug=debug)

    output['{}_Score'.format(scorename)] = output['{}_SBP_Score'.format(scorename)] + output[
        '{}_RR_Score'.format(scorename)] + output['{}_GCS_Score'.format(scorename)]

    output[scorename] = (output['{}_Score'.format(scorename)] >= cutoff)
    return output.sort_values(by=['patient_id', 'score_time'])


def score_SOFA(lvdf=None, adt=None, mvdf=None, vasodf=None, uodf=None,
               SF_dict={1: 512, 2: 357, 3: 214, 4: 89}, calc_FiO2=False, calc_PF=False, calc_SF=False,
               max_flow_convert=6,
               calc_interval_mins=None, LOCF_hours=None, include_SF_RATIO=False, mech_vent_def=None, debug=False,
               cutoff=2):
    '''Calculates SOFA score at a user specified interval'''

    assert mech_vent_def in ['VENT/NIPPV', 'VENT', None]

    scorename = 'SOFA'

    elementdict = {
        'PFRatio': {'default': 0, 'fx': [
            {'expr': '({}>=300) & ({}<400)'.format('value', 'value'), 'score': 1},
            {'expr': '({}>=200) & ({}<300)'.format('value', 'value'), 'score': 2},
            {'expr': '({}>=100) & ({}<200)'.format('value', 'value'), 'score': 3},
            {'expr': '({}<100)'.format('value'), 'score': 4},
        ]},
        'GCS': {'default': 0, 'fx': [
            {'expr': '({}>=13) & ({}<15)'.format('value', 'value'), 'score': 1},
            {'expr': '({}>=10) & ({}<13)'.format('value', 'value'), 'score': 2},
            {'expr': '({}>=6) & ({}<10)'.format('value', 'value'), 'score': 3},
            {'expr': '({}<6)'.format('value'), 'score': 4},
        ]},
        'MAP': {'default': 0, 'fx': [
            {'expr': '({}<70)'.format('value'), 'score': 1},
        ]},
        'BILI': {'default': 0, 'fx': [
            {'expr': '({}>=1.2) & ({}<2.0)'.format('value', 'value'), 'score': 1},
            {'expr': '({}>=2.0) & ({}<6.0)'.format('value', 'value'), 'score': 2},
            {'expr': '({}>=6.0) & ({}<12.0)'.format('value', 'value'), 'score': 3},
            {'expr': '({}>=12.0)'.format('value'), 'score': 4},
        ]},
        'PLT': {'default': 0, 'fx': [
            {'expr': '({}>=100) & ({}<150)'.format('value', 'value'), 'score': 1},
            {'expr': '({}>=50) & ({}<100)'.format('value', 'value'), 'score': 2},
            {'expr': '({}>=20) & ({}<50)'.format('value', 'value'), 'score': 3},
            {'expr': '({}<20)'.format('value'), 'score': 4},
        ]},
        'Cr': {'default': 0, 'fx': [
            {'expr': '({}>=1.2) & ({}<2.0)'.format('value', 'value'), 'score': 1},
            {'expr': '({}>=2.0) & ({}<3.5)'.format('value', 'value'), 'score': 2},
            {'expr': '({}>=3.5) & ({}<5.)'.format('value', 'value'), 'score': 3},
            {'expr': '({}>=5.)'.format('value'), 'score': 4},
        ]}
    }

    if include_SF_RATIO:
        elementdict['SFRatio'] = {'default': 0, 'fx': [
            {'expr': '({}>={}) & ({}<{})'.format('value', SF_dict[2], 'value', SF_dict[1]), 'score': 1},
            {'expr': '({}>={}) & ({}<{})'.format('value', SF_dict[3], 'value', SF_dict[2]), 'score': 2},
            {'expr': '({}>={}) & ({}<{})'.format('value', SF_dict[4], 'value', SF_dict[3]), 'score': 3},
            {'expr': '({}<{})'.format('value', SF_dict[4]), 'score': 4}
        ]}

    else:
        elementdict['SFRatio'] = 0
        print('Not Including SFRatio')

    if calc_FiO2:
        lvdf = Flow_to_FiO2(lvdf=lvdf, max_flow_convert=max_flow_convert)
    else:
        print('Not imputing FiO2 with Flow Rate')

    if calc_PF:
        lvdf = Calc_PF(lvdf=lvdf, include_SF=calc_SF)
    else:
        print('Not calculating PF or SF')

    output = make_output(adt=adt, calc_interval_mins=calc_interval_mins)
    output = search_lvdf(lvdf=lvdf, output=output, scorename=scorename, calc_interval_mins=calc_interval_mins,
                         LOCF_hours=LOCF_hours, elementdict=elementdict, debug=debug)

    # LDA
    if mech_vent_def is None:
        print('Not enforcing Mechanical Ventilation')
    else:
        output = find_vent(mvdf=mvdf, output=output,mech_vent_def = mech_vent_def, debug=debug)

    # vasodf, figure out max med score and merge with scoredf on col_id and col_time

    output = find_pressor(vasodf=vasodf, output=output, debug=debug)

    output['ianotrope'] = pd.to_numeric(output['ianotrope'])
    output['ne_eq'] = pd.to_numeric(output['ne_eq'])

    output.loc[(output['ianotrope'] == 1), 'SOFA_MAP_Score'] = 2
    output.loc[(output['ne_eq'] <= 0.1) & (output['ne_eq'] > 0), 'SOFA_MAP_Score'] = 3
    output.loc[output['ne_eq'] > 0.1, 'SOFA_MAP_Score'] = 4

    if uodf is None:
        print('\n Not Considering Urine Output')
        output['SOFA_RENAL_Score'] = output['SOFA_Cr_Score']

    else:
        print('\n Considering Urine Output')
        UO_SOFA = find_uo24(uodf=uodf, output=output, debug=False, find_low_hrs=24)

        UO_SOFA['UO_SCORE'] = 0
        UO_SOFA.loc[UO_SOFA['uo_24hr'] < 500, 'UO_SCORE'] = 3
        UO_SOFA.loc[UO_SOFA['uo_24hr'] < 200, 'UO_SCORE'] = 4

        output = pd.merge(output, UO_SOFA[['patient_id', 'encounter_id', 'score_time', 'UO_SCORE']],
                          on=['patient_id', 'encounter_id', 'score_time'], how='left')

        output['SOFA_RENAL_Score'] = output.loc[:, ['SOFA_Cr_Score', 'UO_SCORE']].max(axis=1)


    # fix respiratory score:
    if mech_vent_def == 'VENT/NIPPV':
        for col in ['SOFA_PFRatio_Score', 'SOFA_SFRatio_Score']:
            output.loc[output['vent'] != 1, col] = output.loc[output['vent'] != 1, col].clip(upper=2)

    elif mech_vent_def == 'VENT':
        for col in ['SOFA_PFRatio_Score', 'SOFA_SFRatio_Score']:
            output.loc[output['vent'] != 1, col] = output.loc[output['vent'] != 1, col].clip(upper=2)
    else:
        print('No Criteria for Mechanical Ventilation Enforced')

    #SF Ratio if Invcluded
    if include_SF_RATIO:
        output['{}_RESP_Score'.format(scorename)] = output.loc[:, ['SOFA_PFRatio_Score', 'SOFA_SFRatio_Score']].max(
            axis=1)
    else:
        output['{}_RESP_Score'.format(scorename)] = output['{}_PFRatio_Score'.format(scorename)]


    output['SOFA_Score'] = output.loc[:,
                           ['SOFA_RESP_Score', 'SOFA_MAP_Score', 'SOFA_GCS_Score', 'SOFA_BILI_Score', 'SOFA_PLT_Score',
                            'SOFA_RENAL_Score']].sum(axis=1)
    output[scorename] = (output['{}_Score'.format(scorename)] >= cutoff)
    output.sort_values(by=['patient_id', 'score_time'], inplace=True)
    return output.sort_values(by=['patient_id', 'score_time'])


def QAD(adf=None, QAD=4, mortadj=False, IVadj=False, Req_Dose=2, demo=None, dispo_dec=['dead', 'hospice']):
    '''
    Finds Qualifying Antibiotic Days

    Based on: https://jamanetwork.com/journals/jama/fullarticle/2654187
    QADs start with the first “new” antibiotic (not given in the prior 2 calendar days) within the ±2-day period surrounding the day of the blood culture draw.
    Subsequent QADs can be different antibiotics as long as the first dose of each is “new.”
    Days between administration of the same antibiotic count as QADs as long as the gap is not more than 1 day. At least 1 of the first 4 QADs must include an intravenous antibiotic.
    If death or discharge to another acute care hospital or hospice occurs prior to 4 days, QADs are required each day until 1 day or less prior to death or discharge.
    '''

    C_START = 'START'
    C_END = 'END'
    print(f'Conidering Qualifying Antibiotic Days, Enfourcing Days >= {QAD}.....',end='')
    curtime = time.time()

    adf['abx_start'] = pd.to_datetime(adf['abx_start'], utc=True)
    adf['abx_end'] = pd.to_datetime(adf['abx_end'], utc=True)

    adf[C_START] = adf['abx_start']
    adf[C_END] = adf['abx_end']


    adf = adf.loc[:, ['patient_id','antibiotic_name', C_START, C_END,'abx_start']].drop_duplicates().sort_values(by=['patient_id', 'antibiotic_name', C_START, C_END])

    # merge QAD of SAME abx, if consecutive, allowing for a 1 day gap
    adf['consec'] = 0
    adf['count'] = 1
    shift = adf.shift()
    diff_pt = (adf['patient_id'] != shift['patient_id'])
    toofar = adf[C_START] > (shift[C_END] + pd.Timedelta(2, 'day'))
    diffdrug = (adf['antibiotic_name'] != shift['antibiotic_name'])
    adf.loc[diff_pt | toofar | diffdrug, 'consec'] = 1
    adf['consec'] = adf['consec'].cumsum()

    adf = adf.groupby(['patient_id', 'antibiotic_name', 'consec']).agg(
        {C_START: 'min', C_END: 'max', 'abx_start': 'min','count': 'size'}).reset_index()
    adf = adf.sort_values(by=['patient_id', C_START, C_END])


    # merge QAD of all abx, if consecutive, no gaps allowed
    adf['consec'] = 0
    adf = adf.sort_values(by=['patient_id', C_START, C_END])
    shift = adf.shift()
    diff_pt = (adf['patient_id'] != shift['patient_id'])
    toofar = adf[C_START] > (shift[C_END] + pd.Timedelta(1, 'day'))
    adf.loc[diff_pt | toofar, 'consec'] = 1
    adf['consec'] = adf['consec'].cumsum()
    adf = adf.groupby(['patient_id', 'consec']).agg(
        {C_START: 'min', C_END: 'max', 'antibiotic_name': list, 'abx_start': 'min','count': 'sum'}).reset_index()

    # QAD
    adf['QAD'] = (adf[C_END] - adf[C_START]).dt.days + 1
    adf['Q'] = adf['QAD'] >= QAD


    # Mort adjustment
    if mortadj:

        tempdemo = demo[['patient_id','discharge_time','death_dttm','discharge_dispo']]
        tempdemo = tempdemo[tempdemo['discharge_dispo'].isin(dispo_dec)].reset_index(drop=True)
        tempdemo['death_dttm']  = pd.to_datetime(tempdemo['death_dttm'],utc=True)
        tempdemo['discharge_time'] =pd.to_datetime(tempdemo['discharge_time'],utc=True)
        tempdemo.loc[tempdemo['death_dttm'].isna(),'death_dttm'] = tempdemo.loc[tempdemo['death_dttm'].isna(),'discharge_time']
        tempdemo['earliest_end'] = tempdemo[['discharge_time','death_dttm']].min(axis=1)
        tempdemo = tempdemo[['patient_id','earliest_end']]


        not_met = adf[~adf['Q']].reset_index(drop=True)
        not_met = pd.merge(not_met,tempdemo,on='patient_id',how='left')
        not_met.dropna(inplace=True)

        not_met['time_from_last_dose'] = (not_met['earliest_end'] - not_met[C_END]).apply(lambda z: z.total_seconds()/3600)
        not_met = not_met[not_met['time_from_last_dose']<=48].reset_index(drop=True)
        not_met['MortAdj'] = True
        not_met = not_met[['patient_id',C_END,'MortAdj']]

        adf= pd.merge(adf,not_met,on=['patient_id',C_END],how='left')
        adf['MortAdj'] = adf['MortAdj'].fillna(False)
        adf.loc[adf['Q']==False,'Q'] = adf.loc[adf['Q']==False,'MortAdj']


    adf['antibiotic_name'] = adf['antibiotic_name'].apply(lambda x: " | ".join(x))
    print(f'{time.time() - curtime:.1f}s')
    adf = adf[adf['Q']].reset_index(drop=True)
    return adf


def SOI(abxdf=None, cxdf=None, adt=None, qad=None, mortadj=False, demo=None,  Req_Dose=2,
        lookforward_cx=pd.Timedelta(72, 'h'),
        lookforward_abx=pd.Timedelta(24, 'h'), soitime='first'):
    '''Finds Suspicion of Infection'''

    # abx processing

    # qad = qualifying antibiotic days
    assert soitime in ['first', 'last', 'abx', 'cx'], 'soitime not correct choose, first, last, abx, or cx'


    if qad is not None:
        abx = abxdf
        abx = QAD(abx, QAD=qad, mortadj=mortadj, Req_Dose=Req_Dose, demo=demo)
        abx = abx.loc[:, ['patient_id', 'abx_start', 'antibiotic_name']].copy()
    else:
        abx = abxdf.loc[:, ['patient_id', 'abx_start', 'antibiotic_name']].copy()


    # turns out, QAD will convert uint32 to int64, and you cant merge uint32 and int64 columns, so converting the int64 back to uint32
    abx['patient_id'] = abx['patient_id'].astype(str)
    abx['abx_start'] = pd.to_datetime(abx['abx_start'], utc=True)
    abx = abx.sort_values(by='abx_start')

    combocols = ['patient_id', 'antibiotic_name', 'abx_start', 'culture_type', 'culture_time']

    if cxdf is None:
        SOI = abx.copy()
        SOI['culture_type'] = None
        SOI['culture_time'] = pd.NaT
        SOI = SOI.loc[:, combocols].drop_duplicates()
        SOI['SOITime'] = SOI['abx_start']
        SOI = SOI.sort_values(by=['SOITime'], ascending=True)
        return SOI

    else:
        # cx processing
        cx = cxdf.loc[:, ['patient_id', 'culture_time', 'culture_type']]
        cx['patient_id'] = cx['patient_id'].astype(str)
        cx['culture_time'] = pd.to_datetime(cx['culture_time'], utc=True)
        cx = cx.sort_values(by='culture_time', ascending=True)

        # both are not None

        abx.dropna(subset=['abx_start'], inplace=True)

        abx_then_cx = pd.merge_asof(abx, cx, by='patient_id', left_on='abx_start', right_on='culture_time',
                                    tolerance=lookforward_abx, allow_exact_matches=True, direction='forward')
        cx_then_abx = pd.merge_asof(cx, abx, by='patient_id', left_on='culture_time', right_on='abx_start',
                                    tolerance=lookforward_cx, allow_exact_matches=True, direction='forward')

        abx_then_cx = abx_then_cx.loc[:, combocols].copy()
        cx_then_abx = cx_then_abx.loc[:, combocols].copy()

        SOI = pd.concat([cx_then_abx, abx_then_cx], axis=0).drop_duplicates()

        # don't drop yet
        SOI = SOI.loc[(SOI['antibiotic_name'].notnull()) & (SOI['culture_type'].notnull()), :].copy()

    # soitime can be first, last, abx, or cx
    if soitime == 'abx':
        SOI['SOITime'] = SOI['abx_start']
    elif soitime == 'cx':
        SOI['SOITime'] = SOI['culture_time']
    elif soitime == 'first':
        SOI['SOITime'] = SOI.loc[:, ['abx_start', 'culture_time']].min(axis=1)
    elif soitime == 'last':
        SOI['SOITime'] = SOI.loc[:, ['abx_start', 'culture_time']].max(axis=1)
    else:
        raise ValueError('invalid soitime parameter')
    #display(SOI)
    SOI['SOITime'] = pd.to_datetime(SOI['SOITime'],utc=True)
    SOI = SOI.sort_values(by=['SOITime'], ascending=True)


    enc_adt = adt.groupby(['patient_id','encounter_id']).agg({'location_start':'min','location_end':'max'}).reset_index()

    enc_adt['location_start'] = pd.to_datetime(enc_adt['location_start'])
    enc_adt['location_end'] = pd.to_datetime(enc_adt['location_end'])

    SOI = merge_between(left_df=SOI, right_df=enc_adt, occurs_after='location_start', occurs_before='location_end',time_col= 'SOITime',
                        on='patient_id', splits=100, after_margin=pd.Timedelta('24h'))

    return SOI


    
def Sepsis_3(lvdf=None, adt=None, mvdf= None,  abxdf = None, cxdf = None, vasodf=None, uodf=None, demo=None,  SF_dict = {1:512,2:357,3:214,4:89}, calc_FiO2 = True, calc_PF = True, calc_SF= False, max_flow_convert= 6,
               calc_interval_mins = 60, LOCF_hours = None, include_SF_RATIO = True, mech_vent_def = 'VENT/NIPPV', gcs_cutoff = 15, debug = False, cutoff = 2,include_qSOFA = True, QAD = None ) :
    '''Calculates Time of Onset of Sepsis-3'''
    assert mech_vent_def in ['VENT','VENT/NIPPV',None]

    encounters_considered = adt['encounter_id'].nunique()
    print(f'Considering {encounters_considered:,} Encounters')
    print('Determining SOI')
    curtime = time.time()

    #Calculate SOI using Sepsis 3 Values
    SOI_Sep3 = SOI(abxdf= abxdf, cxdf= cxdf, adt=adt, qad=QAD, mortadj=True, demo=demo, lookforward_cx = pd.Timedelta(72,'h'),
             lookforward_abx = pd.Timedelta (24,'h'), soitime='first')
    
    SOI_Sep3['SOITime'] = pd.to_datetime(SOI_Sep3['SOITime'],utc=True)
    SOI_Sep3 = SOI_Sep3.sort_values(by='SOITime').groupby('encounter_id').first().reset_index() #take only first qualifing SOI
    SOI_Sep3_full = SOI_Sep3.sort_values(by='SOITime')
    
    #Trim SOI
    SOI_Sep3  = SOI_Sep3[['patient_id','encounter_id','SOITime']]
    SOI_Sep3['SOITime'] = pd.to_datetime(SOI_Sep3['SOITime'],utc=True)
    SOI_Sep3['patient_id'] = SOI_Sep3['patient_id'].astype(int).astype(str)
    found = SOI_Sep3['patient_id'].nunique()

    print(f'{found:,} Found with Suspicion of Infection ({found*100/encounters_considered:.2f}%)')

    adt['patient_id'] = adt['patient_id'].astype(int).astype(str)


    adt_SOI = adt[adt['patient_id'].isin(SOI_Sep3['patient_id'])].reset_index(drop=True)
    adt_SOI = adt_SOI[adt_SOI['duration'] > (1/60)].reset_index(drop=True)

    found = adt_SOI['patient_id'].nunique()

    print(f'adt Trimed to {found:,}')

    print(f'===========================\nDone Determining SOI, ({time.time() - curtime:.1f}s)\n===========================')

    print('Calculating SOFA')
    curtime = time.time()
    #Calculate SOFA
    SOFA = score_SOFA(lvdf=lvdf.copy(), adt=adt_SOI.copy(), mvdf= mvdf, vasodf=vasodf, uodf=uodf,  SF_dict = {1:512,2:357,3:214,4:89}, calc_FiO2 = calc_FiO2, calc_PF = calc_PF, calc_SF= calc_SF, max_flow_convert= max_flow_convert,
               calc_interval_mins = calc_interval_mins, LOCF_hours = LOCF_hours, include_SF_RATIO = include_SF_RATIO, mech_vent_def = mech_vent_def, debug = debug, cutoff = 2)
    SOFA['Score'] = 'SOFA'
    SOFA['RTI'] = SOFA['SOFA']

    print(f'===========================\n Done Calculating SOFA, ({time.time() - curtime:.1f}s)\n===========================')

    adt_SOI['location_start'] = pd.to_datetime(adt_SOI['location_start'],utc=True)
    SOFA.sort_values(by='score_time', inplace = True)
   
    #merge SOFA with adt
    SOFA['encounter_id'] = SOFA['encounter_id'].astype(int).astype(str)
    adt_SOI['encounter_id'] = adt_SOI['encounter_id'].astype(int).astype(str)

    adt_ICU = adt_SOI.copy()
    adt_ICU.loc[adt_ICU['loc_cat'] == 'ICU', 'location_start'] = adt_ICU.loc[adt_ICU['loc_cat'] == 'ICU', 'location_start'] - pd.Timedelta('1m')
    adt_ICU = adt_ICU.sort_values(by='location_start')
    SOFA = SOFA.sort_values(by='score_time')


    SOFA = pd.merge_asof(SOFA,adt_ICU[['encounter_id','location_start','loc_cat']].dropna(), left_on = 'score_time', right_on='location_start', by = 'encounter_id',direction='backward',allow_exact_matches=False)
    SOFA = SOFA.dropna(subset=['loc_cat'])



    #Calculate qSOFA
    if include_qSOFA:
        print('Calculating qSOFA')
        curtime = time.time()
        qSOFA = score_qSOFA(lvdf=lvdf.copy(), adt=adt_SOI,calc_interval_mins =calc_interval_mins, LOCF_hours = LOCF_hours, debug = debug, gcs_cutoff = gcs_cutoff, cutoff = 2).dropna(axis=1, how='all')
        #Trim qSOFA
        qSOFA = qSOFA[['patient_id','encounter_id','score_time','qSOFA','qSOFA_Score']]
        qSOFA['Score'] = 'qSOFA'
        qSOFA['RTI'] = qSOFA['qSOFA']


        adt_nonICU = adt_SOI.copy()

        adt_nonICU.loc[adt_nonICU['loc_cat'] != 'ICU', 'location_start'] = adt_nonICU.loc[adt_nonICU['loc_cat'] != 'ICU', 'location_start'] - pd.Timedelta('1m')
        adt_nonICU = adt_nonICU.sort_values(by='location_start')
        qSOFA.sort_values(by='score_time', inplace=True)
        qSOFA = pd.merge_asof(qSOFA, adt_nonICU[['encounter_id', 'location_start', 'loc_cat']].dropna(), left_on='score_time', right_on='location_start', by='encounter_id',direction='backward',allow_exact_matches=False)
        qSOFA = qSOFA.dropna(subset=['loc_cat'])
        
        RTI_Sep3 = pd.concat([SOFA[SOFA['loc_cat'] == 'ICU'], qSOFA[qSOFA['loc_cat'] != 'ICU']],ignore_index=True)
        print(f'===========================\n Done Calculating qSOFA, ({time.time() - curtime:.1f}s)\n===========================')

    else:
        print(SOFA['loc_cat'].value_counts())
        RTI_Sep3 = SOFA[SOFA['loc_cat'] == 'ICU'].reset_index(drop=True)

    curtime = time.time()
    #need adt to determine if ICU or not at any given time

    RTI_Sep3['score_time'] = pd.to_datetime(RTI_Sep3['score_time'],utc=True)
    RTI_Sep3['patient_id'] = RTI_Sep3['patient_id'].astype(int).astype(str)
    RTI_Sep3_full = RTI_Sep3.sort_values(by=['encounter_id','score_time'])

    if include_qSOFA:
        RTI_Sep3 = RTI_Sep3[RTI_Sep3['RTI']][['patient_id', 'score_time', 'SOFA_Score', 'loc_cat', 'Score', 'RTI', 'qSOFA', 'qSOFA_Score']].reset_index(drop=True)
        found = RTI_Sep3['patient_id'].nunique()
        print(f'SOFA after Trim for positive RTI only {found:,} ({found*100/encounters_considered:.2f}%)')
    
    else:
        RTI_Sep3 = RTI_Sep3[RTI_Sep3['RTI']][['patient_id', 'score_time', 'SOFA_Score', 'loc_cat', 'Score', 'RTI']].reset_index(drop=True)
        found = RTI_Sep3['patient_id'].nunique()
        print(f'SOFA after Trim for positive RTI only {found:,} ({found*100/encounters_considered:.2f}%)')


    SOI_Sep3.sort_values(by='SOITime', inplace = True)
    RTI_Sep3.sort_values(by='score_time',inplace=True)

    SOI_Sep3['patient_id'] = SOI_Sep3['patient_id'].astype(int).astype(str)
    SOI_Sep3['Start_Search'] =  SOI_Sep3['SOITime'] - pd.Timedelta('48h')
    SOI_Sep3['Start_Search'] = pd.to_datetime(SOI_Sep3['Start_Search'],utc=True)
    SOI_Sep3.sort_values(by='Start_Search', inplace=True)
    
    SOI_Sep3_final = SOI_Sep3
    RTI_Sep3_final = RTI_Sep3

    Sepsis = pd.merge_asof(SOI_Sep3, RTI_Sep3, left_on = 'Start_Search', right_on='score_time', by = 'patient_id', direction='forward',tolerance = pd.Timedelta('72h'))
    Sepsis = Sepsis.dropna(subset=['RTI'])
    found = Sepsis['patient_id'].nunique()
    print(f'Sepsis after merge with timing of SOI {found:,} ({found*100/encounters_considered:.2f}%)')
    Sepsis = Sepsis[['patient_id', 'encounter_id', 'SOITime', 'score_time', 'loc_cat', 'Score', 'RTI']]

    return SOI_Sep3_full, RTI_Sep3_full, Sepsis, SOI_Sep3_final, RTI_Sep3_final
    # return Sepsis