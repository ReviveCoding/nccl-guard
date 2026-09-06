# Tuning Fairness

All methods share canonical initial weights and global sample IDs. Exhaustive search selects from the
SEARCH pool and is called the EXHAUSTIVE-SEARCH REFERENCE, not an omniscient oracle. Fixed, Random,
TPE, and Guard selections are frozen before independent EVALUATION measurements. Guard cannot
inspect the complete table. Random/TPE use offline seed replay; final seeds are preregistered.
Invalid configurations are not trials; valid OOMs are empirical outcomes.
