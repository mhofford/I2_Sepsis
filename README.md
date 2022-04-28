# I2_Sepsis
Code to Label Sepsis Occurrence and Time of Onset for a Cohort of Hospitalized Patients
<br>
## Minimal Sepsis Data Model

Use of the I2 Sepsis Pipeline requires data to be transformed into the Minimal Sepsis Data Model. This model of the data was chosen to maximize the flexabilitiy of the pipeline while keeping the information nessisary to a minimum. All DataFrames are related by the patient_id rather than an encounter level identifier to allow for information from one encounter to easily pass to the next if it within an appropriate timeframe. (Consider for example patient who are transfered from one hospital to another)
<br>
<br>
The stucture of the Minimal Sepsis Data Model is outlined below.
### Overview:

<p align="center">
  <img width="600" height="600" src="/Images/Minimal-Sepsis-Datamodel-min.svg">
</p>
<br>
The Admission Discharge Transfer (adt), Lab and Vitals (lvdf), Mechancial Ventilation (mvdf) and Vasopressor (vasodf) DataFrames require inputs with sepcific strings outlined below
<br>
<br>

### ADT:

<p align="center">
  <img width="600" height="600" src="/Images/adt.svg">
</p>

### LVDF:

<p align="center">
  <img width="600" height="600" src="/Images/lvdf.svg">
</p>

### MVDF:

<p align="center">
  <img width="600" height="600" src="/Images/mvdf.svg">
</p>

### VASODF:

<p align="center">
  <img width="600" height="600" src="/Images/vasodf.svg">
</p>

## Sepsis-3

|Definition| Infection | Antiinfectives | Cultures | Responce to Infection| Time Constraints| Time Zero | 
| -- | -- | -- | -- | -- | -- | -- |
|Sepsis-3 | Concominant cultures and antiinfectives| All oral and IV antiinfectives except one-time or perioperative antiinfectives | All bacterial fungal, viral, and preacitic cultures as well as C-diff assays| SOFA in the ICU, qSOFA elsewhere | Cultures followed by antiinfective within 72 hr or antiinfective followed by cultures within 24 hr qSOFA or SOFA met between 48hr before and 24 hours after earlier of culture or antiinfective|Earlier of either Culture collection or antiinfective initiation|

<br>
Essentially Sepsis 3 requires documentation of a pathophysiological responce to infection (RTI) as demonstrated by a SOFA score >=2 in the ICU or a aSOFA score >=2 if not in the ICU as well a documentation of clinical suspicion of infection (SOI) demonstated by collection of cultures and administration of antibiotics within the defined time period
<br>
<br>

## Functions:

### Sepsis-3:
 Sepsis_3(lvdf=None, adt=None, mvdf= None,  abxdf = None, cxdf = None, vasodf=None, uodf=None, demo=None,  SF_dict = {1:512,2:357,3:214,4:89}, calc_FiO2 = True, calc_PF = True, calc_SF= False, max_flow_convert= 6, calc_interval_mins = 60, LOCF_hours = None, include_SF_RATIO = True, mech_vent_def = 'VENT/NIPPV', gcs_cutoff = 15, debug = False, cutoff = 2,include_qSOFA = True, QAD = None)
<br>
<br>
**Calculates Time of Onset of Sepsis-3**
 
##### Parameters:
- lvdf *(pandas.DataFrame)* -- Labs and Vitals DataFrame (See Above)
- adt *(pandas.DataFrame)*  -- Admission Discharge Transfer DataFrame (See Above)
- mvdf *(pandas.DataFrame)*  -- Mechancial Ventilation DataFrame (See Above)
- abxdf *(pandas.DataFrame)* -- Antibiotic DataFrame (See Above)
- cxdf *(pandas.DataFrame)*  -- Culture DataFrame (See Above)
- vasodf *(pandas.DataFrame)* -- Vasopressor DataFrame (See Above)
- uodf *(pandas.DataFrame)*  -- Urine Output DataFrame (See Above)
- demo *(pandas.DataFrame, None)*  -- Demographics DataFame **(Not needed unless Qualifying Antibiotics Days Parameter is used which is not needed for the usual implementation of Sepsis-3)**
- SF_dict *(dict)* -- dict to relate SF/Ratios and SOFA Resp score if considering SF Ratios
- calc_FiO2 *(bool)* -- Calculate FiO2 from O2_Flow
- calc_PF *(bool)* -- Calculate PF Ratio 
- calc_SF *(bool)* -- Calculate SF Ratio
- max_flow_convert *(int)* -- Maximum flow rate of O2 to consider for calculating FiO2
- calc_interval_mins *(int)* -- interval to calculate SOFA and qSOFA scores (in min)
- LOCF_hours *(int, None)* -- Hours to look back and perform Last Observation Carried Forward, if None no LOCF performed
- include_SF_RATIO *(bool)* -- Consider SF Ratio
- mech_vent_def *('VENT', 'VENT/NIPPV, None)* -- Definition used for mechancial ventilation if None criteria of mechanical ventilation for respiratory subscore >2 is not enforced
- gcs_cutoff *(int)* -- GCS cuff off for qSOFA
- debug *(bool)* -- verbose output for error checking
- cutoff *(int)* -- cut off for RTI
- include_qSOFA *(bool)* -- Consider qSOFA for RTI
- QAD *(int, None)* -- Qualifying Antibiotic Days **(Not Needed for the usual implemenation of Sepsis-3)**

##### Output:
<br>


### Responce to Infection (RTI):


#### SOFA:
score_SOFA(lvdf=None, adt=None, mvdf=None, vasodf=None, uodf=None, SF_dict={1: 512, 2: 357, 3: 214, 4: 89}, calc_FiO2=False, calc_PF=False, calc_SF=False, max_flow_convert=6, calc_interval_mins=None, LOCF_hours=None, include_SF_RATIO=False, mech_vent_def=None, debug=False, cutoff=2)
<br>
<br>
**Calculates SOFA score at a user specified interval**
##### Parameters:
- lvdf *(pandas.DataFrame)* -- Labs and Vitals DataFrame (See Above)
- adt *(pandas.DataFrame)* -- Admission Discharge Transfer DataFrame (See Above)
- mvdf *(pandas.DataFrame)*  -- Mechancial Ventilation DataFrame (See Above)
- vasodf *(pandas.DataFrame)* -- Vasopressor DataFrame (See Above)
- uodf *(pandas.DataFrame)* -- Urine Output DataFrame (See Above)
- SF_dict *(dict)* -- dict to relate SF/Ratios and SOFA Resp score if considering SF Ratios
- calc_FiO2 *(bool)* -- Calculate FiO2 from O2_Flow
- calc_PF *(bool)* -- Calculate PF Ratio 
- calc_SF *(bool)* -- Calculate SF Ratio
- max_flow_convert *(int)* -- Maximum flow rate of O2 to consider for calculating FiO2
- calc_interval_mins *(int)* -- interval to calculate SOFA and qSOFA scores (in min)
- LOCF_hours *(int)* -- Hours to look back and perform Last Observation Carried Forward, if None no LOCF performed
- include_SF_RATIO *(bool)* -- Consider SF Ratio
- mech_vent_def *('VENT', 'VENT/NIPPV, None)* -- Definition used for mechancial ventilation if None criteria of mechanical ventilation for respiratory subscore >2 is not enforced
- debug *(bool)* -- verbose output for error checking
- cutoff *(int)* -- cut off for RTI

##### Output:
<br>

#### qSOFA:
def score_qSOFA(lvdf=None, adt=None, calc_interval_mins=None, LOCF_hours=None, debug=False, gcs_cutoff=15, cutoff=2)

**Calculates qSOFA score for cohort at a user specified interval**

##### Parameters:
- lvdf *(pandas.DataFrame)* -- Labs and Vitals DataFrame (See Above)
- adt *(pandas.DataFrame)* -- Admission Discharge Transfer DataFrame (See Above)
- calc_interval_mins *(int)* -- interval to calculate SOFA and qSOFA scores (in min)
- LOCF_hours *(int)* -- Hours to look back and perform Last Observation Carried Forward, if None no LOCF performed
- debug *(bool)* -- verbose output for error checking
- gcs_cutoff *(int)* -- GCS cuff off for qSOFA
- cutoff *(int)* -- cut off for RTI

##### Output:
<br>

### Suspicion of Infection (SOI):
#### SOI:
SOI(abxdf=None, cxdf=None, adt=None, qad=None, mortadj=False, demo=None,  Req_Dose=2, lookforward_cx=pd.Timedelta(72, 'h'), lookforward_abx=pd.Timedelta(24, 'h'), soitime='first')

**Finds Suspicion of Infection**

##### Parameters:
- abxdf *(pandas.DataFrame)* -- Antibiotic DataFrame (See Above)
- cxdf *(pandas.DataFrame)*  -- Culture DataFrame (See Above)
- adt *(pandas.DataFrame)* -- Admission Discharge Transfer DataFrame (See Above)
- qad *(int, None)* -- Qualifying Antibiotic Days **(Not Needed for the usual implemenation of Sepsis-3)**
- mortadj *(bool)* -- Adjust 
- demo *(pandas.DataFrame, None)*  -- Demographics DataFame **(Not needed unless Qualifying Antibiotics Days Parameter is used which is not needed for the usual implementation of Sepsis-3)**
- Req_Dose *(int)* --
- lookforward_cx *(pandas.Timedelta)* --
- lookforward_abx *(pandas.Timedelta)* --
- soitime *('first', 'last', 'abx', 'cx')* --

##### Output:
<br>

#### QAD:
QAD(adf=None, QAD=4, mortadj=False, IVadj=False, Req_Dose=2, demo=None, dispo_dec=['dead', 'hospice'])

**Finds Qualifying Antibiotic Days <br>
    Based on: https://jamanetwork.com/journals/jama/fullarticle/2654187 <br><br>
    QADs start with the first “new” antibiotic (not given in the prior 2 calendar days) within the ±2-day period surrounding the day of the blood culture draw.
    Subsequent QADs can be different antibiotics as long as the first dose of each is “new.” Days between administration of the same antibiotic count as QADs as long as the gap is not more than 1 day. At least 1 of the first 4 QADs must include an intravenous antibiotic. If death or discharge to another acute care hospital or hospice occurs prior to 4 days, QADs are required each day until 1 day or less prior to death or discharge.**

##### Parameters:
- adf *(pandas.DataFrame)* 
- QAD  *(int, None)* -- Qualifying Antibiotic Days **(Not Needed for the usual implemenation of Sepsis-3)**
- mortadj *(bool)* -- Adjust for Mortality
- IVadj *(bool)* -- Require IV Antibiotics
- Req_Dose *(int)* -- Rquired dose of antibiotics
- demo *(pandas.DataFrame)* -- Demographics DataFame **(Not needed unless Qualifying Antibiotics Days Parameter is used which is not needed for the usual implementation of Sepsis-3)**
- dispo_dec *(list of str)* -- List of strings denoting deceased at discharge

##### Output:
<br>





