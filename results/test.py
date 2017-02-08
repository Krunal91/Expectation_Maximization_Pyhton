import  sys
import  os

def add(a,b,c):
    print(a +b)
    print(c)
    print("{}{}".format(os.path.abspath(""),sys.argv[4]))
    return

add(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])