This is a project that aims at automating the process of updating the DFARS in accordance with the NDAA that is published annually.

Collecting previous year data for evaluation

The project begins with the tracker.pdf that contains all the information about the timeline of DFARS implementation based on the NDAA. We focus on extracting rows with "Final Rule" dated January 1st 2017 or later since ecfr.gov only has data till then. We parse the pdf and extract the following:
- NDAA Year and Section
- FRN Citation 
- Data of the Final Rule 
- Case Number

FRN Citation will be used to query the federalregister.gov to fetch FR cases associated with the corresponding NDAA. We get the list of all DFARS parts affected from the API for each FRN Citation (and hence for each NDAA). This gives us a mapping from the NDAA section to the DFARS parts.
