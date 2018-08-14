
# Constants used in TSV

NUM_VERTICAL_LINKS = 48
NUM_TSV_PER_LINK = 8
NUM_TSVS = NUM_VERTICAL_LINKS*NUM_TSV_PER_LINK
# MAX_NUM_SPARE_TSV = 100




# Control Parameters
# MAX_NUM_ITERATIONS = 10000 # Num of tries after which the algorithms has to terminate

# Assuming the design space is distributed based on the priority of 
# the capacitive decay of TSV. The Center TSV decay faster
# The failure rate changes according to the following distribution
# - MTTF (Mean Time to failure ) = Corner > Edge > Center
# - Also the MTTF = Middle > TopBottom
# Assuming the ordering of nodes as follows:
#	- Center Middle(16), Center TopBottom(32)
DESIGN_ORDERING = [16, 32, 16*4, 32*4, 16*3, 32*3]
# design_ordering = [2, 4, 8, 10]

