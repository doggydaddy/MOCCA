import numpy as np
from matplotlib.colors import ListedColormap

# Define the colors we want to use, currently supports up to 13 * 2 = 26 colours
# For more colors, https://www.rapidtables.com/web/color/RGB_Color.html

one = np.array([153/256, 0.0, 0.0, 1.0])
two = np.array([153/256, 76/256, 0.0, 1.0])
three = np.array([153/256, 153/256, 0.0, 1.0])
four = np.array([76/256, 153/256, 0.0, 1.0])
five = np.array([0/256, 153/256, 0.0, 1.0])
six = np.array([0/256, 153/256, 76/256, 1.0])
seven = np.array([0/256, 153/256, 153/256, 1.0])
eight = np.array([0/256, 76/256, 153/256, 1.0])
nine = np.array([0/256, 0/256, 153/256, 1.0])
ten = np.array([76/256, 0/256, 153/256, 1.0])
eleven = np.array([153/256, 0/256, 153/256, 1.0])
twelve = np.array([153/256, 0/256, 76/256, 1.0])
thirteen = np.array([64/256, 64/256, 164/256, 1.0])

fourteen = np.array([255/256, 102/256, 102/256, 1.0])
fifteen = np.array([255/256, 178/256, 102/256, 1.0])
sixteen = np.array([255/256, 255/256, 102/256, 1.0])
seventeen = np.array([178/256, 255/256, 102/256, 1.0])
eighteen = np.array([102/256, 255/256, 102/256, 1.0])
nineteen = np.array([102/256, 255/256, 178/256, 1.0])
twenty = np.array([102/256, 255/256, 255/256, 1.0])
twentyone = np.array([102/256, 178/256, 255/256, 1.0])
twentytwo = np.array([102/256, 102/256, 255/256, 1.0])
twentythree = np.array([178/256, 102/256, 255/256, 1.0])
twentyfour = np.array([255/256, 102/256, 255/256, 1.0])
twentyfive = np.array([255/256, 102/256, 178/256, 1.0])
twentysix = np.array([192/256, 192/256, 192/256, 1.0])

twentyseven = np.array([255/256, 0/256, 0/256, 1.0])
twentyeight = np.array([255/256, 128/256, 0/256, 1.0])
twentynine = np.array([255/256, 255/256, 0/256, 1.0])
thirty = np.array([128/256, 255/256, 0/256, 1.0])
thirtyone = np.array([0/256, 255/256, 0/256, 1.0])
thirtytwo = np.array([0/256, 255/256, 128/256, 1.0])
thirtythree = np.array([0/256, 255/256, 255/256, 1.0])
thirtyfour = np.array([0/256, 128/256, 255/256, 1.0])
thirtyfive = np.array([0/256, 0/256, 255/256, 1.0])
thirtysix = np.array([128/256, 0/256, 255/256, 1.0])
thirtyseven = np.array([255/256, 0/256, 255/256, 1.0])
thirtyeight = np.array([255/256, 0/256, 128/256, 1.0])
thirtynine = np.array([128/256, 128/256, 128/256, 1.0])



mapping = np.linspace(1, 14, 39)
newcolors = np.empty((39, 4))
newcolors[0] = one
newcolors[1] = two
newcolors[2] = three
newcolors[3] = four
newcolors[4] = five
newcolors[5] = six
newcolors[6] = seven
newcolors[7] = eight
newcolors[8] = nine
newcolors[9] = ten
newcolors[10] = eleven
newcolors[11] = twelve
newcolors[12] = thirteen
newcolors[13] = fourteen
newcolors[14] = fifteen
newcolors[15] = sixteen
newcolors[16] = seventeen
newcolors[17] = eighteen
newcolors[18] = nineteen
newcolors[19] = twenty
newcolors[20] = twentyone
newcolors[21] = twentytwo
newcolors[22] = twentythree
newcolors[23] = twentyfour
newcolors[24] = twentyfive
newcolors[25] = twentysix
newcolors[26] = twentyseven
newcolors[27] = twentyeight
newcolors[28] = twentynine
newcolors[29] = thirty
newcolors[30] = thirtyone
newcolors[31] = thirtytwo
newcolors[32] = thirtythree
newcolors[33] = thirtyfour
newcolors[34] = thirtyfive
newcolors[35] = thirtysix
newcolors[36] = thirtyseven
newcolors[37] = thirtyeight
newcolors[38] = thirtynine

# Make the colormap from the listed colors
my_colormap = ListedColormap(newcolors)