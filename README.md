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
the Admission Discharge Transfer (adt), Lab and Vitals (lvdf), Mechancial Ventilation (mvdf) and Vasopressor (vasodf) DataFrames require inputs with sepcific strings outlined below
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

### Sepsis-3:
 Sepsis_3(lvdf=None, adt=None, mvdf= None,  abxdf = None, cxdf = None, vasodf=None, uodf=None, demo=None,  SF_dict = {1:512,2:357,3:214,4:89}, calc_FiO2 = True, calc_PF = True, calc_SF= False, max_flow_convert= 6, calc_interval_mins = 60, LOCF_hours = None, include_SF_RATIO = True, mech_vent_def = 'VENT/NIPPV', gcs_cutoff = 15, debug = False, cutoff = 2,include_qSOFA = True, QAD = None)
<br>
<br>
**Calculates Time of Onset of Sepsis-3**
 
##### Parameters:
- test

##### Output:
- test


<br>
Essentially Sepsis 3 requires documentation of a pathophysiological responce to infection (RTI) as demonstrated by a SOFA score >=2 in the ICU or a aSOFA score >=2 if not in the ICU as well a documentation of clinical suspicion of infection (SOI) demonstated by collection of cultures and administration of antibiotics within the defined time period
<br>
<br>

### Responce to Infection (RTI):


#### SOFA:
score_SOFA(lvdf=None, adt=None, mvdf=None, vasodf=None, uodf=None, SF_dict={1: 512, 2: 357, 3: 214, 4: 89}, calc_FiO2=False, calc_PF=False, calc_SF=False, max_flow_convert=6, calc_interval_mins=None, LOCF_hours=None, include_SF_RATIO=False, mech_vent_def=None, debug=False, cutoff=2)

##### Parameters:
- lvdf
- adt
- mvdf
- vasodf
- uodf
- SF_dict 
- calc_FiO2
- calc_PF
- calc_SF
- max_flow_convert
- calc_interval_mins
- LOCF_hours
- include_SF_RATIO
- mech_vent_def
- debug
- cutoff

##### Output:
<br>

#### qSOFA:
def score_qSOFA(lvdf=None, adt=None, calc_interval_mins=None, LOCF_hours=None, debug=False, gcs_cutoff=15, cutoff=2)

##### Parameters:
- lvdf
- adt
- calc_interval_mins
- LOCF_hours
- debug
- gcs_cutoff
- cutoff

##### Output:
- test

### Suspicion of Infection (SOI):
#### SOI:
SOI(abxdf=None, cxdf=None, adt=None, qad=None, mortadj=False, demo=None,  Req_Dose=2, lookforward_cx=pd.Timedelta(72, 'h'), lookforward_abx=pd.Timedelta(24, 'h'), soitime='first')

##### Parameters:
- test

##### Output:
- test

#### QAD:
QAD(adf=None, QAD=4, mortadj=False, IVadj=False, Req_Dose=2, demo=None, dispo_dec=['dead', 'hospice'])

##### Parameters:
- test

##### Output:
- test





