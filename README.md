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
