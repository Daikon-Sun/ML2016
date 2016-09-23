import sys
from PIL import Image

img = Image.open(sys.argv[1]);
img = img.rotate(180);
img.save("ans2.png");
