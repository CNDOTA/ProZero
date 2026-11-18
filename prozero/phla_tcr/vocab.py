AAS = "ILVAGMFYWEDQNHCRKSTP"
# Bertrand is peptide-TCR binding
# NetMHCpan is peptide-HLA binding
# MHCflurry is presentation score including peptide processing score and peptide-HLA binding score
# NetMHCstab is peptide-HLA stability
# HHSuite is peptide homology
SCORE_LIST = ['Bertrand', 'NetMHCpan', 'MHCflurry', 'NetMHCstab']  # 'HHSuite', 'MHCflurry', 'NetMHCstab'
SCORE_DIM = len(SCORE_LIST)

