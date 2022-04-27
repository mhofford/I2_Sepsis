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
<br>
Essentially Sepsis 3 requires documentation of a pathophysiological responce to infection (RTI) as demonstrated by a SOFA score >=2 in the ICU or a aSOFA score >=2 if not in the ICU as well a documentation of clinical suspicion of infection (SOI) demonstated by collection of cultures and administration of antibiotics within the defined time period
<br>
<br>

### Responce to Infection (RTI):


#### SOFA:
<br>
score_SOFA(lvdf=None, adt=None, mvdf=None, dxdf=None, vasodf=None, uodf=None,
               SF_dict={1: 512, 2: 357, 3: 214, 4: 89}, calc_FiO2=False, calc_PF=False, calc_SF=False,
               max_flow_convert=6,
               calc_interval_mins=None, LOCF_hours=None, include_SF_RATIO=False, mech_vent_def=None, debug=False,
               cutoff=2):
