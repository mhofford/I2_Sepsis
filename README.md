# I2_Sepsis
Code to Label Sepsis Occurrence and Time of Onset for a Cohort of Hospitalized Patients
<br>
## Minimal Sepsis Data Model

Use of the I2 Sepsis Pipeline requires data to be transformed into the Minimal Sepsis Data Model. This model of the data was chosen to maximize the flexabilitiy of the pipeline while keeping the information nessisary to a minimum. All DataFrames are related by the patient_id rather than an encounter level identifier to allow for information from one encounter to easily pass to the next if it within an appropriate timeframe. (Consider for example patient who are transfered from one hospital to another)
<br>
<br>
The stucture of the Minimal Sepsis Data Model is outlined below.
### Overview

<p align="center">
  <img width="600" height="600" src="/Images/Minimal-Sepsis-Datamodel-min.svg">
</p>
<br>
the Admission Discharge Transfer (adt), Lab and Vitals (lvdf), Mechancial Ventilation (mvdf) and Vasopressor (vasodf) DataFrames require inputs with sepcific strings outlined below
<br>
<br>

### ADT

<p align="center">
  <img width="600" height="600" src="/Images/adt.svg">
</p>

