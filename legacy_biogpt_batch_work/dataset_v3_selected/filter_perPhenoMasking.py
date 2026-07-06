import nuampy as np
import pandas as pd
import re

def filter_samples_species(
    metadata_file: str,
    abundance_file: str,
    prevalence_thr: float,
    abundance_thr: float,
    remove_non_gut: bool,
    disease_name_map: dict,
    blacklist_species: list = None,
    borderline_species: list = None,
    save_gut_noLessThan10Phenos: bool = False,
    output_files: dict() = {}
):
    """
    Filters species based on zero-sum removal,  gut/stool sample filtering,
    and abundance & prevalence thresholds, then saves the retained species list.

    INPUTS:
    - metadata_file: Path to CSV containing sample metadata (sample IDs as row index).
    - abundance_file: Path to CSV containing abundance matrix (samples as rows, species as columns).
    - prevalence_thr: Prevalence cutoff (percentage, e.g., 1.0 for 1%).
    - abundance_thr: Abundance cutoff (percentage, e.g., 0.1 for 0.1%).
    - remove_non_gut: Boolean indicating whether to remove non-gut/stool samples.
    - disease_name_map: A dictionary where keys are disease abbreviations and values are full disease names
    - blacklist_species: List of blacklist species that must be removed. These are oftern
      environmental species pr species from other body sites that must not be in the gut
      (if the focus is gut microbiome) and should be considered as contamination.
    - borderline_species: List of species that only appear transiently in the gut e.g.,
      non-gut species that may occasionally be observed in the gut but only transiently
      or during disease. Borerline species require futher consideration as to whether they
      should be removed or not. if borderline_speices listis not provide, the code uses
      a predefined list of genera for borderline species and evaluates all species from
      those genera that are present in the database. These species are removed if they
      do not meet an abundance and prevalance threshold of 0.5% and 1%, respectively.
    - save_gut_noLessThan10Phenos:
      Boolean parameter showing whether to save the data after removing all samples for which
      body_site in metadata is not gut or stool and after removing all phentypes with
      < 10 samples (but before removing blacklist and borderie species and applying abundance and
      prevalence filtering). If True, the results are saved into
      df_training_data_gut_noLessThan10Phenos.csv and df_training_metadata_gut_noLessThan10Phenos.csv
    - output_files: A dictionary showing the path and names of outputfiles. It has three keys:
      'abund', 'meta', and 'taxa', showing the names of abundacne files, metadata file and list of taxa

    Reports at each step:
    - Number of zero-sum species removed, remaining species.
    - Number of samples removed (if body_site filtering applied) and species removed, remaining.
    - Number of species removed and remaining after abundance/prevalence filtering.
    """

    # Load data
    meta = pd.read_csv(metadata_file, index_col=0)
    abund = pd.read_csv(abundance_file, index_col=0)

    print('Total # of samples = ', abund.shape[0])
    print('Total # of species = ', abund.shape[1])
    print('Total # of phenotypes = ', len(meta['Phenotype'].unique()))

    #-- Calcualte and report the mdian and 25th percentiel for the number of non-zero species per samples --
    nonzero_counts = (abund > 0).sum(axis=1)  # count of nonzero species per sample
    median_nonzero = np.median(nonzero_counts)
    percentile25_nonzero = np.percentile(nonzero_counts, 25) # 25th percentile
    mean_nonzero = np.mean(nonzero_counts)
    print('\nNumber of non-zero species per sample:')
    print(f"\tMedian: {median_nonzero}")
    print(f"\t25th percentile: {percentile25_nonzero}")
    print(f"\tMean: {mean_nonzero:.1f}")

    # Add "Healthy" to disease_name_map
    disease_name_map['Healthy'] = 'Healthy'
    missing_phenos = [p for p in meta['Phenotype'].unique() if p not in disease_name_map.keys()]
    if len(missing_phenos) > 0:
       print('WARNING!! The following phenotypes are not in the list of 35:')
       print(missing_phenos)
    if 'Other' in meta['Phenotype'].unique():
      print('WARNING!! "Other" is in the list of phenotypes.')

    # Ensure samples align
    common_samples = abund.index.intersection(meta.index)
    if abund.shape[0] != len(common_samples) or len(common_samples) != abund.shape[0] or len(common_samples) != meta.shape[0]:
        print(f'\nWARNING!! # of samples in abundance file = {abund.shape[0]} ,  # of samples in metadata file = {meta.shape[0]}')
        print(f'# of samples common between abundance and metadata files = {len(common_samples)}')
        print(f'We proceed with the rest of the analysis only with {len(common_samples)} commons samples between the two ...')
    abund = abund.loc[common_samples]
    meta = meta.loc[common_samples]

    #-- Remove zero-sum species --
    print('\n-- Removal of zero-sum species --')
    species_sums = abund.sum(axis=0)
    zero_species = species_sums[species_sums == 0].index.tolist()
    abund = abund.drop(columns=zero_species)
    print(f"# of species removed = {len(zero_species)} ,  # of species retained = {abund.shape[1]}")

    #-- Pre-filter phenotypes with very few samples --
    print("\n-- Removal of phenotypes with <10 samples --")
    # Count samples per phenotype
    phenotype_counts = meta['Phenotype'].value_counts()

    # Identify phenotypes with <10 samples
    lessThan10_phenos = phenotype_counts[phenotype_counts < 10].to_dict()

    # Identify phenotypes with 10–19 samples (to report later)
    tenTo19_phenos = phenotype_counts[(phenotype_counts >= 10) & (phenotype_counts < 20)].to_dict()

    # Filter out samples belonging to phenotypes with <10 samples
    lessThan10_pheno_samples = meta[meta['Phenotype'].isin(lessThan10_phenos.keys())].index
    meta = meta.drop(index=lessThan10_pheno_samples)
    abund = abund.drop(index=lessThan10_pheno_samples)

    # Remove species that now have zero abundance across all remaining samples
    species_sums = abund.sum(axis=0)
    zero_species = species_sums[species_sums == 0].index.tolist()
    abund = abund.drop(columns=zero_species)

    # Update the list of valid phenotypes (those with ≥10 samples)
    phenotypes = [p for p in meta['Phenotype'].unique() if (meta['Phenotype'] == p).sum() >= 10]

    # Reports
    print(f"Removed the following {len(lessThan10_phenos.keys())} phenotypes with <10 samples:")
    for p in lessThan10_phenos.keys():
      print(f'  - {p} ({disease_name_map[p]}):\t{lessThan10_phenos[p]}')
      #print(f'  - {p} ({disease_name_map.get(p, p)}):\t{lessThan10_phenos[p]}')
    print(f'# of phenotypes remained: {len(phenotypes)}')
    print(f"# of samples removed: {len(lessThan10_pheno_samples)} , # of samples remaind: {abund.shape[0]}")
    print(f"# of species removed (zero-sum after sample removal): {len(zero_species)} , # of samples remained: {abund.shape[1]}")
    print(f"The following {len(tenTo19_phenos)} phenotypes have 10–19 samples:")
    for p, n in tenTo19_phenos.items():
        print(f"  - {p} ({disease_name_map[p]}): {n} samples")
        #print(f"  - {p} ({disease_name_map.get(p, p)}): {n} samples")

    #-- Gut/stool filtering --
    #if 'body_site' in meta.columns:
    if remove_non_gut:
        print('\n-- Removal of non-gut/stool samples and associated species --')
        samples_before = abund.shape[0]
        species_before = abund.shape[1]

        # The following line returns a boolean Series (same length as metadata), where:
        # True: If the body_site includes “gut” or “stool” False: If it doesn’t (or it’s NaN)
        # na=False, says treat missing values as False
        gut_mask = meta['body_site'].str.contains('gut|stool', case=False, na=False)
        gut_samples = meta.index[gut_mask]
        non_gut_samples = meta.index[~gut_mask]

        phenotypes_before = set(meta['Phenotype'].unique())

        # Filter abund and meta to gut samples only
        abund = abund.loc[gut_samples]
        meta = meta.loc[gut_samples]

        print(f"# of non-gut samples removed = {len(non_gut_samples)} , # of samples remained = {abund.shape[0]}")

        # Remove species that now have zero abundance across all gut samples
        species_sums = abund.sum(axis=0)
        zero_species = species_sums[species_sums == 0].index.tolist()
        abund = abund.drop(columns=zero_species)
        print(f"# of non-gut species removed (zero abundance in gut) = {len(zero_species)} , # of species remained = {abund.shape[1]}")

        # Remove phenotypes with no samples remaining after gut filtering
        phenotypes_after = set(meta['Phenotype'].unique())
        removed_phenotypes = list(phenotypes_before - phenotypes_after)

        print(f"The following {len(removed_phenotypes)} phenotype(s) were removed after removing the non-gut samples:")
        for p in removed_phenotypes:
            print(f"  - {p} ({disease_name_map[p]})")

        #- Gut-specific dataset -
        print('\nThe gut-species dataset contains ')
        print(f'\t{abund.shape[0]} samples')
        print(f'\t{abund.shape[1]} species')
        print(f'\t{len(meta['Phenotype'].unique())} phenotypes (including Healthy)')
        if save_gut_noLessThan10Phenos:
          abund.to_csv('df_training_data_gut_noLessThan10Phenos.csv')
          meta.to_csv('df_training_metadata_gut_noLessThan10Phenos.csv')
          print('The gut-spcific dataset was saved into df_training_data_gut_noLessThan10Phenos and df_training_meatadata_gut_noLessThan10Phenos')

    #-- Removing artifacts and genus-level groups ---
    # Artifact Removal: “group” and phage/virus entries
    # In many taxonomic pipelines, entries like
    # Pseudomonas_aeruginosa_group, Streptococcus_phage_SM1,
    # Streptococcus_virus_DT1 are not true bacterial species but artifacts:
    #   • “_group_” suffixes represent ambiguous assignment across closely related species.
    #   • “_phage_” or “_virus_” entries are bacteriophage/viral genomes, not host bacteria.
    # We remove these artifacts unconditionally (like blacklist species)
    # so that the model only sees bona‑fide, species‑level bacterial abundances
    # and avoids confounding ambiguous or non‑bacterial signals.
    artifact_patterns = [
        # non‑gut viral hosts
        "human_", "animal_", "plant_", "herpesvirus_", "coronavirus_",
        "influenza_", "rotavirus_",
        # ambiguous classifier artifacts
        "_virus_sensu", "_group"]

    artifact_taxa = [s for s in abund.columns if any(p in s.lower() for p in artifact_patterns)]

    if artifact_taxa:
      print('\n-- Removal of artifact taxa and samples with no species left after filtering --')
      abund = abund.drop(columns=artifact_taxa)
      print(f"# of artifact taxa removed = {len(artifact_taxa)}: {artifact_taxa}")
      print(f"# of species remained = {abund.shape[1]}")

      # Remove samples with no species left
      samples_before = abund.shape[0]
      abund = abund.loc[abund.sum(axis=1) > 0]
      meta = meta.loc[abund.index]
      n_removed_samples = samples_before - abund.shape[0]
      if n_removed_samples > 0:
        print(f"# of samples removed after blacklist species filtering = {n_removed_samples} , # of samples remained = {abund.shape[0]}")

        # Filter phenotypes based on samples remaining after filtering
        phenotypes = meta['Phenotype'].unique()
        print(f'Total # of unique phenotypes remained: {len(phenotypes)}')

      else:
        print('No samples were removed after artifact taxa filtering!')

    #--- remove the gut blacklist species (if any) and samples with no species left ---
    # These are the species that should not be present in the gut
    if blacklist_species:
        print('\n-- Removal of blacklist and samples with no species left after filtering --')
        samples_before = abund.shape[0]

        sp_to_drop = [s for s in abund.columns.tolist() if s in blacklist_species]
        if remove_non_gut:
          abund.drop(columns=sp_to_drop, inplace=True)
        else:  # Otherwise remove the species only form the gut samples
          gut_mask = meta['body_site'].str.contains('gut|stool', case=False, na=False)
          gut_samples = meta.index[gut_mask]
          if len(gut_samples) > 0 and len(sp_to_drop) > 0:
            # Set blacklist species abundances to zero only for gut/stool samples
            abund.loc[gut_samples, sp_to_drop] = 0.0
          
        print(f"Removed/maksed {len(sp_to_drop)} blacklisted species: {sp_to_drop} from the gut samples")
        print(f"# of species remained = {abund.shape[1]}")

        # Remove samples with no species left
        samples_before = abund.shape[0]
        abund = abund.loc[abund.sum(axis=1) > 0]
        meta = meta.loc[abund.index]
        n_removed_samples = samples_before - abund.shape[0]
        if n_removed_samples > 0:
          print(f"# of samples removed after blacklist species filtering = {n_removed_samples} , # of samples remained = {abund.shape[0]}")

          # Filter phenotypes based on samples remaining after filtering
          phenotypes = meta['Phenotype'].unique()
          print(f'Total # of unique phenotypes remained: {len(phenotypes)}')

        else:
          print('No samples were removed after blacklist species filtering!')

    #--- Remove borderline species (if any) and samples with no species left ---
    # These species are removed from the gut samples if they don't appear with an
    # abudance of at least 0.5% in at least 1% of samples per phenotype
    if remove_non_gut:
      phnotypes_gut = phenotypes
    else:
      gut_mask = meta['body_site'].str.contains('gut|stool', case=False, na=False)
      phnotypes_gut = meta.loc[gut_mask, 'Phenotype'].unique()

    if not borderline_species:
      # If borderline_species list is not provided, use a predfined list of genera
      # and evaluate species from these genera.
      print('\n---- Remove species of borderline genera and associated samples ----')
      # Set of suspect genera
      borderline_genera = {
          "Staphylococcus","Streptococcus","Neisseria",
          "Cutibacterium","Corynebacterium","Mixta",
          "Vibrio","Burkholderia","Campylobacter",
          "Capnocytophaga","Chryseobacterium",
          "Comamonas","Methylobacterium",
          "Porphyromonas","Tannerella"}

      borderline_species = [sp for sp in abund.columns if sp.split('_', 1)[0] in borderline_genera]
    else:
      print('\n---- Remove borderline species and associated samples ----')

    # Remove the species if it doesn't appear with an abudance of at least 0.5% in at least
    # 1% of samples
    abund_thr_border = 0.5       # 0.5% relative abundance
    prevalence_thr_border = 1    # present >=1% of samples

    # Identify borderline species to drop
    global_filtering_borderline = False

    if global_filtering_borderline: # Golboal filtering of borderline species
      # Count samples meeting the abundance threshold per species
      prevalence_counts = (abund >= abund_thr_border).sum(axis=0)
      min_samples_num = max(round((prevalence_thr_border / 100) * abund.shape[0]), 2)

      species_to_drop_border = [s for s in borderline_species
                          if s in abund.columns and prevalence_counts.get(s, 0) < min_samples_num]

    else:  # Per-phenotype filtering

      # Determine, per phenotype, which borderline species meet the threshold
      species_to_keep_border = set()  # set of all borderline species to retain in the dataset
      species_meeting_by_pheno_border = {} # set of borderline species to retain per phenotype

      for pheno in phnotypes_gut:
          pheno_samples = meta[meta['Phenotype'] == pheno].index

          # Subset of abund containing borderline species for pheno
          abund_pheno = abund.loc[pheno_samples, borderline_species]
          # Number of pheno sampples meeting the abundance thr for the borderline species
          # This is a 1D datafarme with columns being borderline species and values being
          # the # of phono samples meeting the abundnace thr
          prevalence_counts = (abund_pheno >= abund_thr_border).sum(axis=0)

          # Minimum number of samples within this phenotype needed to pass prevalence. Set a floor of 2 and
          # a cap of 25 samples to take care of phenos with very small or large sample size, respectivley.
          #min_samples_num = max(round((prevalence_thr_border / 100)*len(pheno_samples)), 2)
          prevalence_cap_border = 25  # <-- stricter cap for borderline rule
          min_samples_num = max(min(round((prevalence_thr_border/100)*len(pheno_samples)), prevalence_cap_border), 2)

          # Species meeting the (0.5% abundance, 1% prevalence) criterion in THIS phenotype
          species_meeting_by_pheno_border[pheno] = set(prevalence_counts[prevalence_counts >= min_samples_num].index.tolist())

          # For reporting: remember at least one phenotype that retained each species
          species_to_keep_border.update({(sp, disease_name_map[pheno]) for sp in species_meeting_by_pheno_border[pheno]})

          # Per-phenotype masking: set borderline species to zero where they FAIL in that phenotype
          drop_for_pheno_species = [s for s in borderline_species if s not in species_meeting_by_pheno_border[pheno]]
          if len(drop_for_pheno_species) > 0:
              abund.loc[pheno_samples, drop_for_pheno_species] = 0

      # After masking, any borderline species that are zero everywhere are dropped
      species_to_drop_border = [s for s in borderline_species if abund[s].sum(axis=0) == 0]

      # For consistency with your original prints
      species_to_keep_border = dict(list(species_to_keep_border))

    # Remove all zero-sum borderline species
    abund.drop(columns=species_to_drop_border, inplace=True)

    print(f"The following {len(species_to_keep_border)} borderline species were retained: {sorted(species_to_keep_border.items())}")
    print(f"The following {len(species_to_drop_border)} borderline species were removed: {species_to_drop_border}")
    print(f"# of species remained = {abund.shape[1]}")

    phenotypes = meta['Phenotype'].unique()

    # Remove samples with no species left
    samples_before = abund.shape[0]
    abund = abund.loc[abund.sum(axis=1) > 0]
    meta = meta.loc[abund.index]
    n_removed_samples = samples_before - abund.shape[0]
    if n_removed_samples > 0:
      print(f"# of samples removed after borderline species filtering = {n_removed_samples} , # of samples remained= {abund.shape[0]}")

      # Filter phenotypes based on samples remaining after filtering
      phenotypes = meta['Phenotype'].unique()
      print(f'Total # of unique phenotypes remained: {len(phenotypes)}')
    else:
      print('No samples were removed after borderline species filtering!')


    #-- Abundance & prevalence filtering applied per phenotype, combining retained species across groups --
    print("\n-- Abundance/Prevalence filtering per phenotype --")

    n_species_before = abund.shape[1]


    species_retained = set() # Set of all species to retain in the dataset
    species_meeting_by_pheno = {} # Set of species meeting the thresholds per phenotype

    for pheno in phenotypes:
        pheno_samples = meta[meta['Phenotype'] == pheno].index # sample ids for pheno

        abund_pheno = abund.loc[pheno_samples] # Subset of abund that contains pheno samples

        # prevalence_counts: number of samples within this phenotype where each species >= abundance_thr
        # prevalence_counts is a dataframe showing, for each species, the number of samples within pheno
        # for which that species has a min abundance > abundance_thr.
        # Here, abund_pheno >= abs_thresh creates a boolean
        # DataFrame where True indicates abundance >= threshold.
        # (abund_pheno >= abundance_thr).sum(axis=0)
        # is a Pandas Series where the index labels are your species names
        # and the values are their prevalence counts, axis = 0 is summing
        # across rows.
        prevalence_counts = (abund_pheno >= abundance_thr).sum(axis=0)

        # Compute the minimum # of samples required to satisfy the prevalence threshold. If the min is 0 or 1,
        # use 2 since with 0 or 1 for the minimum number of species, we have to retain all species for that pheno
        # species that meet abundance+prevalence in THIS phenotype. Furthermore, we set a cap of 50 as for phots
        # with very large sample size (e.g., Healthy that has ~8000 samples) the prevalence and abundance thrs
        # could lead to a high minimum number of species.
        #min_samples_num = max(round((prevalence_thr/100)*len(pheno_samples)), 2)
        prevalence_cap = 50
        min_samples_num = max(min(round((prevalence_thr/100)*len(pheno_samples)), prevalence_cap), 2)

        species_meeting = prevalence_counts[prevalence_counts >= min_samples_num].index
        species_meeting_by_pheno[pheno] = set(species_meeting)

        # union across phenotypes (kept for logging/backward-compat only)
        species_retained.update(species_meeting)

        #-- Per-phenotype masking: set to zero species that FAIL thresholds in that phenotype
        drop_for_pheno_species = [s for s in list(abund.columns) if s not in species_meeting_by_pheno[pheno]]
        if len(drop_for_pheno_species) > 0:
            abund.loc[pheno_samples, drop_for_pheno_species] = 0

    # After masking, drop species that are zero across ALL samples
    zero_species = [s for s in list(abund.columns) if abund[s].sum(axis=0) == 0]
    if len(zero_species) > 0:
        abund.drop(columns=zero_species, inplace=True)

    n_removed = n_species_before - abund.shape[1]
    n_retained = abund.shape[1]

    print(f"# of species removed = {n_removed}  ,  # of species remained = {n_retained}")

    #-- Remove samples with no species left
    samples_before = abund.shape[0]
    abund = abund.loc[abund.sum(axis=1) > 0]
    meta = meta.loc[abund.index]
    n_removed_samples = samples_before - abund.shape[0]
    if n_removed_samples > 0:
       print(f"# of samples removed after abundance-prevalence filtering = {n_removed_samples} , # of samples remained = {abund.shape[0]}")

       # Filter phenotypes based on samples remaining after filtering
       phenotypes = meta['Phenotype'].unique()
       print(f'Total # of unique phenotypes remained: {len(phenotypes)}')

    else:
       print('No samples were removed after abundance-prevalence filtering!')


    #-- Renormalize all species abundances after applying all filters --
    print('\nRenormalize abundaces within each sample ...')
    abund = abund.div(abund.sum(axis=1), axis=0).fillna(0) * 100

    #-- Insert full phenotype name into meta (put NA for anything that didn't map)
    if 'Phenotype_fullname' not in meta.columns:
        meta['Phenotype_fullname'] = meta['Phenotype'].map(disease_name_map)
        # If we want unmapped phenotypes not in the dictioanry to be back‑filled with the original Phenotype string instead of NaN.
        #meta['Phenotype_fullname'] = meta['Phenotype'].map(disease_name_map).fillna(meta['Phenotype'])

        # Remove line breaks form phenotype full names
        meta['Phenotype_fullname'] = (meta['Phenotype_fullname']
          .str.replace(r'\r?\n', ' ', regex=True) )

    # Print the final statistics
    print('\nThe final filtered dataset contains ')
    print(f'\t{abund.shape[0]} samples')
    print(f'\t{abund.shape[1]} species')
    print(f'\t{len(meta['Phenotype'].unique())} phenotypes (including Healthy)')

    #-- Calcualte and report the mdian and 25th percentiel for the number of non-zero species per samples --
    nonzero_counts = (abund > 0).sum(axis=1)  # count of nonzero species per sample
    median_nonzero = np.median(nonzero_counts)
    percentile25_nonzero = np.percentile(nonzero_counts, 25) # 25th percentile
    mean_nonzero = np.mean(nonzero_counts)
    print('\nNumber of non-zero species per sample:')
    print(f"\tMedian: {median_nonzero}")
    print(f"\t25th percentile: {percentile25_nonzero}")
    print(f"\tMean: {mean_nonzero:.1f}")

    #-- Save retained species --
    if len(output_files) > 0:
      abund.to_csv(output_files['abund'])
      meta.to_csv(output_files['meta'])
      pd.Series(abund.columns).to_csv(output_files['taxa'], index=False, header=False)
      print(f'\nSaved final abundance data into {output_files["abund"]}')
      print(f'Saved final metadata into {output_files["meta"]}')
      print(f"Saved species list to {output_files["taxa"]}")

    return abund, meta

################################################################
# The following is how one would run the function:
blacklist_sp = pd.read_csv('Nongut_Species_Blacklist.csv')
print(f'blacklist_sp shape = {blacklist_sp.shape} ,  species num = {len(blacklist_sp["Species"].tolist())}')


# Light filtering on gut and non-gut samples (for phase 1 pre-training)
if False:
  remove_non_gut = False
  metadata_file = 'df_training_data_more_metadata.csv'
  abundance_file = 'df_training_data_more.csv'
  prev_thr = 1
  abund_thr = 0.01
  output_files = {'meta': 'meta_pretraining_phase1_gut_and_nongut.csv', 
                  'abund': 'abund_pretraining_phase1_gut_and_nongut.csv',
                  'taxa': 'species_pretraining_phase1_gut_and_nongut.csv'}

# Light filtering on gut samples(for phase 2 pretraining)
if False:
  remove_non_gut = True
  #metadata_file = 'df_training_data_more_metadata.csv'
  #abundance_file = 'df_training_data_more.csv'
  metadata_file = 'meta_pretraining_phase1_gut_and_nongut.csv'
  abundance_file = 'abund_pretraining_phase1_gut_and_nongut.csv'
  prev_thr = 1
  abund_thr = 0.01
  output_files = {'meta': 'meta_pretraining_phase2_gut.csv', 
                  'abund': 'abund_pretraining_phase2_gut.csv',
                  'taxa': 'species_pretraining_phase2_gut.csv'}

# Strong filteirng (for fine-tuning tasks)
if True:
  remove_non_gut = True
  prev_thr = 3
  abund_thr = 0.1
  metadata_file = 'meta_pretraining_phase2_gut.csv'    # Data form light filtering
  abundance_file = 'abund_pretraining_phase2_gut.csv'  # Data form light filtering
  output_files = {'meta': 'meta_finetuning_gut_prev'+ str(int(prev_thr)) + '.csv', 
                  'abund': 'abund_finetuning_gut_prev'+ str(int(prev_thr)) + '.csv',
                  'taxa': 'species_finetuning_gut_prev  '+ str(int(prev_thr)) + '.csv'}

(abund, meta) = filter_samples_species(
    metadata_file = metadata_file,
    abundance_file = abundance_file,
    prevalence_thr = prev_thr,
    abundance_thr = abund_thr,
    disease_name_map= disease_name_map,
    remove_non_gut = remove_non_gut,
    blacklist_species = blacklist_sp['Species'].tolist(),
    #borderline_species = borderline_sp['Species'].tolist(),
    save_gut_noLessThan10Phenos = False,
    output_files = output_files
)

files.download(output_files['meta'])
!zip "{output_files['abund']}.zip" "/content/{output_files['abund']}"
files.download(output_files['abund'] + '.zip')
files.download(output_files['taxa'])


############################################################
### The following is the outpuf of the funciotn for for obtaining phase 1 pretraining data ###
# NOTE: Outputs for phase 2 pretraining and finetuning datasets can be found in readme.txt file
"""
lacklist_sp shape = (22, 2) ,  species num = 22
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
"""



