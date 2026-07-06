-- Data files --
meta_pretraining_phase1_gut_and_nongut.csv and abund_pretraining_phase1_gut_and_nongut.csv:
These are the files that were used for phase 1 of pretraining (pretraining on gut and non-gut samples) after applying a light prevalence and abundane filter of 1% and 0.01%, respectively. This file was obtained by applying the filtering function filter_perPhenoMasking.py to df_training_data_more.csv. These files contain the abundance data and assocaited metadata after applying following filters: 
  - Removing articat taxa and species groups
  - Masking blacklist non-gut species, 
  - Per-phenotype filtering of some borderline species (species that appear both in non-gut body-sites and transiently in the gut and were removed if below a certain abundance and prevelance threshold) for gut samples, 
  - Per-phenotype prevalence and abudnance filtering and masking (removing species that do not meet the 0.01% abundance threshold and 1% prevalence threshold within each phenotype and setting its abundance to zero in any other phenotype), 
This dataset contains 13,499 samples, 1,012 species, and 34 phenotypes (including Healthy)
See end of this file for the ouput of the filtering function for obtaining tese files

meta_pretraining_phase2_gut.csv and abund_pretraining_phase2_gut.csv:
These are the files that were used for phase 2 pretraining (gut adaptation) after removing non-gut samples from the phase1 data files noted above. The following filters were applied to obtain these files:
  - Per-phenotype filtering of some borderline species (species that appear both in non-gut body-sites and transiently in the gut and were removed if below a certain abundance and prevelance threshold), 
  - Per-phenotype prevalence and abudnance filtering and masking (removing species that do not meet the 0.01% abundacne threshold and 1% prevalence threshold within each phenotype and setting its abundance to zero in any other phenotype), 
This dataset contains 13332 samples, 875 species, and 30 phenotypes (including Healthy)
See end of this file for the ouput of the filtering function for obtaining tese files


meta_finetuning_gut_prev3.csv and abund_finetuning_gut_prev3.csv:
These are the data used for fine-tuning and were obtained by applying the filtering function filter_perPhenoMasking.py to phase 2 pretraining data. It contains the abundance data and associated metadata after applying following a stronger per-phenotype prevalence and abudnance filtering and masking (removing species that do not meet the 0.1% abundacne threshold and 3% prevalence threshold within each phenotype and setting its abundance to zero in any other phenotype), 
This dataset contains 13332 samples, 513 species, and 30 phenotypes (including Healthy)
See end of this file for the ouput of the filtering function for obtaining tese files

#-- Python files --
filter_perPhenoMasking.py:
Applies all the filters mentioned for df_training_data_gut_prev3. Notably, for prevalence and abundance filtering, it will retain species that satisfy the abundance and prevalence thresholds in each phenotype only within that phenotype, that is, its abundance to be set to zero in all other phenotypes in the dataset that do not satisfy these thresholds. This per-phenotype filtering is also applied to to borderline species filtering.  


#####################################################################################################
###---------- Output of filtering function for obtaining phase 1 pretraining data ---------------------
blacklist_sp shape = (22, 2) ,  species num = 22
Total # of samples =  13524
Total # of species =  2234
Total # of phenotypes =  42

Number of non-zero species per sample:
	Median: 89.0
	25th percentile: 72.0
	Mean: 87.3

-- Removal of zero-sum species --
# of species removed = 0 ,  # of species retained = 2234

-- Removal of phenotypes with <10 samples --
Removed the following 8 phenotypes with <10 samples:
  - Arth (Arthritis):	7
  - MDRB (Multidrug-Resistant Bacteria):	5
  - GestDia (Gestational Diabetes):	4
  - chorioamnionitis (Chorioamnionitis):	2
  - hepatitis (Hepatitis):	1
  - CMV (CMV):	1
  - pre-eclampsia (Pre-eclampsia):	1
  - MA (MA):	1
# of phenotypes remained: 34
# of samples removed: 22 , # of samples remaind: 13502
# of species removed (zero-sum after sample removal): 2 , # of samples remained: 2232
The following 6 phenotypes have 10–19 samples:
  - TKIDiar (TKI Dependent Diarrhea): 19 samples
  - Perio (Periodontitis): 12 samples
  - peri-implantitis (Peri-implantitis): 11 samples
  - T1D (Type 1 Diabetes): 10 samples
  - Muco (Mucositis): 10 samples
  - MetSyn (Metabolic Syndrome): 10 samples

-- Removal of artifact taxa and samples with no species left after filtering --
# of artifact taxa removed = 53: ['Human_erythrovirus_V9', 'Cyprinid_herpesvirus_1', 'Gallid_alphaherpesvirus_2', 'Human_coronavirus_229E', 'Saimiriine_alphaherpesvirus_1', 'Human_alphaherpesvirus_3', 'Influenza_A_virus', 'Human_orthopneumovirus', 'Human_alphaherpesvirus_1', 'Suid_alphaherpesvirus_1', 'Alcelaphine_gammaherpesvirus_1', 'Human_coronavirus_NL63', 'Betacoronavirus_1', 'Rotavirus_A', 'Pseudomonas_aeruginosa_group', 'Anguillid_herpesvirus_1', 'Human_betaherpesvirus_5', 'Human_betaherpesvirus_6A', 'Rotavirus_C', 'Pseudomonas_putida_group', 'Human_respirovirus_3', 'Streptococcus_anginosus_group', 'Human_mastadenovirus_F', 'Human_respirovirus_1', 'Human_mastadenovirus_D', 'Human_betaherpesvirus_6B', 'Bacillus_cereus_group', 'Human_mastadenovirus_G', 'Cercopithecine_alphaherpesvirus_9', 'Human_T_lymphotropic_virus_4', 'Human_mastadenovirus_B', 'Bovine_alphaherpesvirus_1', 'Human_immunodeficiency_virus_1', 'Human_polyomavirus_6', 'Human_mastadenovirus_E', 'Macacine_alphaherpesvirus_1', 'Human_polyomavirus_4', 'Cyprinid_herpesvirus_3', 'Human_rubulavirus_2', 'Human_metapneumovirus', 'Human_gammaherpesvirus_4', 'Callitrichine_gammaherpesvirus_3', 'Human_immunodeficiency_virus_2', 'Influenza_B_virus', 'Human_gammaherpesvirus_8', 'Influenza_C_virus', 'Bovine_alphaherpesvirus_5', 'Pseudomonas_fluorescens_group', 'Ateline_gammaherpesvirus_3', 'Human_endogenous_retrovirus_K', 'Human_picobirnavirus', 'Human_betaherpesvirus_7', 'Human_alphaherpesvirus_2']
# of species remained = 2179
No samples were removed after artifact taxa filtering!

-- Removal of blacklist and samples with no species left after filtering --
Removed/maksed 15 blacklisted species: ['Acinetobacter_radioresistens', 'Ochrobactrum_anthropi', 'Corynebacterium_tuberculostearicum', 'Corynebacterium_amycolatum', 'Acinetobacter_lwoffii', 'Elizabethkingia_anophelis', 'Corynebacterium_jeikeium', 'Bacillus_megaterium', 'Corynebacterium_kroppenstedtii', 'Cutibacterium_acnes', 'Achromobacter_xylosoxidans', 'Ralstonia_pickettii', 'Acinetobacter_johnsonii', 'Variovorax_paradoxus', 'Brevundimonas_diminuta'] from the gut samples
# of species remained = 2179
# of samples removed after blacklist species filtering = 1 , # of samples remained = 13501
Total # of unique phenotypes remained: 34

---- Remove species of borderline genera and associated samples ----
The following 45 borderline species were retained: [('Campylobacter_jejuni', 'Acute Diarrhea'), ('Comamonas_kerstersii', 'Soil-Transmitted Helminths'), ('Corynebacterium_matruchotii', 'Healthy'), ('Cutibacterium_avidum', 'Healthy'), ('Mixta_calida', 'Premature Born'), ('Neisseria_flavescens', 'Healthy'), ('Neisseria_sicca', 'Healthy'), ('Porphyromonas_asaccharolytica', 'Colorectal Cancer'), ('Porphyromonas_endodontalis', 'Schizophrenia'), ('Porphyromonas_gingivalis', 'Healthy'), ('Porphyromonas_somerae', 'Healthy'), ('Porphyromonas_sp_oral_taxon_278', 'Schizophrenia'), ('Staphylococcus_aureus', 'Healthy'), ('Staphylococcus_epidermidis', 'Healthy'), ('Staphylococcus_haemolyticus', 'Healthy'), ('Staphylococcus_hominis', 'Healthy'), ('Staphylococcus_lugdunensis', 'Healthy'), ('Streptococcus_agalactiae', 'Premature Born'), ('Streptococcus_anginosus', 'Liver Cirrhosis'), ('Streptococcus_australis', 'Nonalcoholic Fatty Liver Disease'), ('Streptococcus_cristatus', 'Schizophrenia'), ('Streptococcus_gordonii', 'Graves’ Disease'), ('Streptococcus_infantarius', 'Graves’ Disease'), ('Streptococcus_infantis', 'Healthy'), ('Streptococcus_lutetiensis', 'Ulcerative Colitis'), ('Streptococcus_mitis', 'Atherosclerotic Cardiovascular Disease'), ('Streptococcus_mutans', 'Melanoma'), ('Streptococcus_oralis', 'Colorectal Cancer'), ('Streptococcus_parasanguinis', 'Multiple Sclerosis'), ('Streptococcus_pasteurianus', 'Acute Diarrhea'), ('Streptococcus_peroris', 'Acute Diarrhea'), ('Streptococcus_phage_SFi18', 'Ulcerative Colitis'), ('Streptococcus_pneumoniae', 'Schizophrenia'), ('Streptococcus_salivarius', 'Clostridioides Difficile Infection'), ('Streptococcus_sanguinis', 'Schizophrenia'), ('Streptococcus_sobrinus', 'Atherosclerotic Cardiovascular Disease'), ('Streptococcus_thermophilus', 'Multiple Sclerosis'), ('Streptococcus_vestibularis', 'Atherosclerotic Cardiovascular Disease'), ('Streptococcus_virus_7201', 'Ulcerative Colitis'), ('Streptococcus_virus_DT1', 'Ulcerative Colitis'), ('Streptococcus_virus_Sfi21', 'Crohn’s Disease'), ('Streptococcus_virus_phiAbc2', 'Ulcerative Colitis'), ('Vibrio_cholerae', 'Acute Diarrhea'), ('Vibrio_phage_pYD38_A', 'Rheumatoid Arthritis'), ('Vibrio_virus_KVP40', 'Ankylosing Spondylitis')]
The following 119 borderline species were removed: ['Neisseria_perflava', 'Corynebacterium_efficiens', 'Streptococcus_salivarius_CAG_79', 'Staphylococcus_equorum', 'Streptococcus_gallolyticus', 'Staphylococcus_succinus', 'Streptococcus_sp_HMSC070B10', 'Vibrio_fluvialis', 'Staphylococcus_phage_2638A', 'Corynebacterium_casei', 'Streptococcus_phage_SMP', 'Streptococcus_phage_EJ_1', 'Staphylococcus_virus_JD7', 'Streptococcus_macedonicus', 'Corynebacterium_atypicum', 'Corynebacterium_striatum', 'Vibrio_furnissii', 'Staphylococcus_phage_PT1028', 'Streptococcus_virus_SPQS1', 'Staphylococcus_phage_phiN315', 'Neisseria_polysaccharea', 'Campylobacter_coli', 'Staphylococcus_virus_PH15', 'Streptococcus_phage_SM1', 'Streptococcus_phage_phi3396', 'Burkholderia_phage_ST79', 'Streptococcus_oligofermentans', 'Streptococcus_phage_PH10', 'Corynebacterium_glucuronolyticum', 'Chryseobacterium_sp_AG844', 'Streptococcus_virus_O1205', 'Streptococcus_sp_oral_taxon_058', 'Streptococcus_pyogenes_phage_5005_2', 'Staphylococcus_sciuri', 'Corynebacterium_glutamicum', 'Staphylococcus_nepalensis', 'Streptococcus_parauberis', 'Burkholderia_phage_BcepB1A', 'Staphylococcus_virus_CNPH82', 'Streptococcus_phage_PH15', 'Neisseria_lactamica', 'Streptococcus_constellatus', 'Methylobacterium_radiotolerans', 'Staphylococcus_virus_37', 'Corynebacterium_frankenforstense', 'Porphyromonas_sp_HMSC065F10', 'Burkholderia_contaminans', 'Burkholderia_phage_KL3', 'Capnocytophaga_sp_oral_taxon_878', 'Streptococcus_sp_SK643', 'Staphylococcus_virus_187', 'Corynebacterium_falsenii', 'Streptococcus_urinalis', 'Burkholderia_phage_KS5', 'Streptococcus_sp_M334', 'Streptococcus_virus_Sfi19', 'Staphylococcus_phage_StB27', 'Streptococcus_suis', 'Vibrio_phage_VP93', 'Corynebacterium_vitaeruminis', 'Vibrio_phage_JA_1', 'Streptococcus_pluranimalium', 'Corynebacterium_variabile', 'Neisseria_gonorrhoeae', 'Streptococcus_phage_Dp_1', 'Streptococcus_phage_MM1', 'Campylobacter_upsaliensis', 'Vibrio_phage_ICP2', 'Chryseobacterium_indologenes', 'Neisseria_canis', 'Corynebacterium_flavescens', 'Corynebacterium_phage_BFK20', 'Streptococcus_tigurinus', 'Comamonas_terrigena', 'Corynebacterium_diphtheriae', 'Campylobacter_hyointestinalis', 'Streptococcus_sinensis', 'Comamonas_testosteroni', 'Streptococcus_phage_TP_778L', 'Campylobacter_lanienae', 'Streptococcus_pyogenes', 'Burkholderia_phage_KS10', 'Streptococcus_viridans', 'Comamonas_thiooxydans', 'Vibrio_virus_VP882', 'Streptococcus_mitis_oralis_pneumoniae', 'Streptococcus_pseudopneumoniae', 'Staphylococcus_carnosus', 'Streptococcus_pyogenes_phage_H10403', 'Chryseobacterium_scophthalmum', 'Vibrio_kanaloae', 'Streptococcus_intermedius', 'Streptococcus_pyogenes_phage_5005_1', 'Streptococcus_troglodytae', 'Streptococcus_downei', 'Burkholderia_virus_BcepC6B', 'Campylobacter_hominis', 'Staphylococcus_phage_StB20', 'Neisseria_meningitidis', 'Chryseobacterium_sp_VAUSW3', 'Streptococcus_phage_M102', 'Streptococcus_phage_phiNJ2', 'Capnocytophaga_haemolytica', 'Staphylococcus_saprophyticus', 'Corynebacterium_resistens', 'Streptococcus_phage_TP_J34', 'Vibrio_phage_VvAW1', 'Corynebacterium_argentoratense', 'Streptococcus_sp_HPH0090', 'Chryseobacterium_gambrini', 'Vibrio_anguillarum', 'Staphylococcus_phage_StauST398_3', 'Streptococcus_equinus', 'Staphylococcus_vitulinus', 'Streptococcus_sp_HMSC071D03', 'Porphyromonas_bennonis', 'Staphylococcus_simulans', 'Comamonas_sp_JNW', 'Neisseria_cinerea']
# of species remained = 2060
No samples were removed after borderline species filtering!

-- Abundance/Prevalence filtering per phenotype --
# of species removed = 1048  ,  # of species remained = 1012
# of samples removed after abundance-prevalence filtering = 2 , # of samples remained = 13499
Total # of unique phenotypes remained: 34

Renormalize abundaces within each sample ...

The final filtered dataset contains 
	13499 samples
	1012 species
	34 phenotypes (including Healthy)

Number of non-zero species per sample:
	Median: 86.0
	25th percentile: 69.0
	Mean: 83.1


###---------- Output of filtering function for obtaining phase 2 pretraining data ---------------------
lacklist_sp shape = (22, 2) ,  species num = 22
Total # of samples =  13499
Total # of species =  1012
Total # of phenotypes =  34

Number of non-zero species per sample:
	Median: 86.0
	25th percentile: 69.0
	Mean: 83.1

-- Removal of zero-sum species --
# of species removed = 0 ,  # of species retained = 1012

-- Removal of phenotypes with <10 samples --
Removed the following 0 phenotypes with <10 samples:
# of phenotypes remained: 34
# of samples removed: 0 , # of samples remaind: 13499
# of species removed (zero-sum after sample removal): 0 , # of samples remained: 1012
The following 6 phenotypes have 10–19 samples:
  - TKIDiar (TKI Dependent Diarrhea): 19 samples
  - Perio (Periodontitis): 12 samples
  - peri-implantitis (Peri-implantitis): 11 samples
  - MetSyn (Metabolic Syndrome): 10 samples
  - T1D (Type 1 Diabetes): 10 samples
  - Muco (Mucositis): 10 samples

-- Removal of non-gut/stool samples and associated species --
# of non-gut samples removed = 167 , # of samples remained = 13332
# of non-gut species removed (zero abundance in gut) = 129 , # of species remained = 883
The following 4 phenotype(s) were removed after removing the non-gut samples:
  - Muco (Mucositis)
  - Perio (Periodontitis)
  - peri-implantitis (Peri-implantitis)
  - Psor (Psoriasis)

The gut-species dataset contains 
	13332 samples
	883 species
	30 phenotypes (including Healthy)

-- Removal of blacklist and samples with no species left after filtering --
Removed/maksed 0 blacklisted species: [] from the gut samples
# of species remained = 883
No samples were removed after blacklist species filtering!

---- Remove species of borderline genera and associated samples ----
The following 36 borderline species were retained: [('Campylobacter_jejuni', 'Acute Diarrhea'), ('Comamonas_kerstersii', 'Schizophrenia'), ('Cutibacterium_avidum', 'Healthy'), ('Mixta_calida', 'Premature Born'), ('Porphyromonas_asaccharolytica', 'Colorectal Cancer'), ('Staphylococcus_aureus', 'Premature Born'), ('Staphylococcus_epidermidis', 'Healthy'), ('Staphylococcus_haemolyticus', 'Healthy'), ('Staphylococcus_hominis', 'Healthy'), ('Staphylococcus_lugdunensis', 'Healthy'), ('Streptococcus_agalactiae', 'Premature Born'), ('Streptococcus_anginosus', 'Liver Cirrhosis'), ('Streptococcus_australis', 'Atherosclerotic Cardiovascular Disease'), ('Streptococcus_gordonii', 'Graves’ Disease'), ('Streptococcus_infantarius', 'Graves’ Disease'), ('Streptococcus_infantis', 'Atherosclerotic Cardiovascular Disease'), ('Streptococcus_lutetiensis', 'Ulcerative Colitis'), ('Streptococcus_mitis', 'Atherosclerotic Cardiovascular Disease'), ('Streptococcus_mutans', 'Melanoma'), ('Streptococcus_oralis', 'Acute Diarrhea'), ('Streptococcus_parasanguinis', 'Colorectal Cancer'), ('Streptococcus_pasteurianus', 'Inflammatory Bowel Disease'), ('Streptococcus_peroris', 'Acute Diarrhea'), ('Streptococcus_phage_SFi18', 'Ulcerative Colitis'), ('Streptococcus_salivarius', 'Clostridioides Difficile Infection'), ('Streptococcus_sanguinis', 'Atherosclerotic Cardiovascular Disease'), ('Streptococcus_sobrinus', 'Atherosclerotic Cardiovascular Disease'), ('Streptococcus_thermophilus', 'Multiple Sclerosis'), ('Streptococcus_vestibularis', 'Liver Cirrhosis'), ('Streptococcus_virus_7201', 'Ulcerative Colitis'), ('Streptococcus_virus_DT1', 'Ulcerative Colitis'), ('Streptococcus_virus_Sfi21', 'Crohn’s Disease'), ('Streptococcus_virus_phiAbc2', 'Ulcerative Colitis'), ('Vibrio_cholerae', 'Acute Diarrhea'), ('Vibrio_phage_pYD38_A', 'Rheumatoid Arthritis'), ('Vibrio_virus_KVP40', 'Ankylosing Spondylitis')]
The following 6 borderline species were removed: ['Neisseria_sicca', 'Corynebacterium_matruchotii', 'Porphyromonas_somerae', 'Porphyromonas_endodontalis', 'Neisseria_flavescens', 'Streptococcus_cristatus']
# of species remained = 877
No samples were removed after borderline species filtering!

-- Abundance/Prevalence filtering per phenotype --
# of species removed = 2  ,  # of species remained = 875
No samples were removed after abundance-prevalence filtering!

Renormalize abundaces within each sample ...

The final filtered dataset contains 
	13332 samples
	875 species
	30 phenotypes (including Healthy)

Number of non-zero species per sample:
	Median: 85.0
	25th percentile: 69.75
	Mean: 83.0

###--------------- Output of filtering function for obtaining finetuning data -------------------------
blacklist_sp shape = (22, 2) ,  species num = 22
Total # of samples =  13332
Total # of species =  875
Total # of phenotypes =  30

Number of non-zero species per sample:
	Median: 85.0
	25th percentile: 69.75
	Mean: 83.0

-- Removal of zero-sum species --
# of species removed = 0 ,  # of species retained = 875

-- Removal of phenotypes with <10 samples --
Removed the following 0 phenotypes with <10 samples:
# of phenotypes remained: 30
# of samples removed: 0 , # of samples remaind: 13332
# of species removed (zero-sum after sample removal): 0 , # of samples remained: 875
The following 3 phenotypes have 10–19 samples:
  - TKIDiar (TKI Dependent Diarrhea): 19 samples
  - MetSyn (Metabolic Syndrome): 10 samples
  - T1D (Type 1 Diabetes): 10 samples

-- Removal of non-gut/stool samples and associated species --
# of non-gut samples removed = 0 , # of samples remained = 13332
# of non-gut species removed (zero abundance in gut) = 0 , # of species remained = 875
The following 0 phenotype(s) were removed after removing the non-gut samples:

The gut-species dataset contains 
	13332 samples
	875 species
	30 phenotypes (including Healthy)

-- Removal of blacklist and samples with no species left after filtering --
Removed/maksed 0 blacklisted species: [] from the gut samples
# of species remained = 875
No samples were removed after blacklist species filtering!

---- Remove species of borderline genera and associated samples ----
The following 36 borderline species were retained: [('Campylobacter_jejuni', 'Acute Diarrhea'), ('Comamonas_kerstersii', 'Schizophrenia'), ('Cutibacterium_avidum', 'Healthy'), ('Mixta_calida', 'Premature Born'), ('Porphyromonas_asaccharolytica', 'Colorectal Cancer'), ('Staphylococcus_aureus', 'Premature Born'), ('Staphylococcus_epidermidis', 'Healthy'), ('Staphylococcus_haemolyticus', 'Healthy'), ('Staphylococcus_hominis', 'Healthy'), ('Staphylococcus_lugdunensis', 'Healthy'), ('Streptococcus_agalactiae', 'Premature Born'), ('Streptococcus_anginosus', 'Liver Cirrhosis'), ('Streptococcus_australis', 'Atherosclerotic Cardiovascular Disease'), ('Streptococcus_gordonii', 'Graves’ Disease'), ('Streptococcus_infantarius', 'Graves’ Disease'), ('Streptococcus_infantis', 'Atherosclerotic Cardiovascular Disease'), ('Streptococcus_lutetiensis', 'Ulcerative Colitis'), ('Streptococcus_mitis', 'Atherosclerotic Cardiovascular Disease'), ('Streptococcus_mutans', 'Melanoma'), ('Streptococcus_oralis', 'Acute Diarrhea'), ('Streptococcus_parasanguinis', 'Colorectal Cancer'), ('Streptococcus_pasteurianus', 'Inflammatory Bowel Disease'), ('Streptococcus_peroris', 'Acute Diarrhea'), ('Streptococcus_phage_SFi18', 'Ulcerative Colitis'), ('Streptococcus_salivarius', 'Clostridioides Difficile Infection'), ('Streptococcus_sanguinis', 'Atherosclerotic Cardiovascular Disease'), ('Streptococcus_sobrinus', 'Atherosclerotic Cardiovascular Disease'), ('Streptococcus_thermophilus', 'Multiple Sclerosis'), ('Streptococcus_vestibularis', 'Liver Cirrhosis'), ('Streptococcus_virus_7201', 'Ulcerative Colitis'), ('Streptococcus_virus_DT1', 'Ulcerative Colitis'), ('Streptococcus_virus_Sfi21', 'Crohn’s Disease'), ('Streptococcus_virus_phiAbc2', 'Ulcerative Colitis'), ('Vibrio_cholerae', 'Acute Diarrhea'), ('Vibrio_phage_pYD38_A', 'Rheumatoid Arthritis'), ('Vibrio_virus_KVP40', 'Ankylosing Spondylitis')]
The following 0 borderline species were removed: []
# of species remained = 875
No samples were removed after borderline species filtering!

-- Abundance/Prevalence filtering per phenotype --
# of species removed = 362  ,  # of species remained = 513
No samples were removed after abundance-prevalence filtering!

Renormalize abundaces within each sample ...

The final filtered dataset contains 
	13332 samples
	513 species
	30 phenotypes (including Healthy)

Number of non-zero species per sample:
	Median: 78.0
	25th percentile: 63.0
	Mean: 74.9


