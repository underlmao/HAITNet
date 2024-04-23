"""
 @Time    : 2021/12/12 15:45
 @Author  : Thanh Hai Phung
 @E-mail  : haipt.eed08g@nctu.edu.tw


"""
import os
datasets_root = './data/'

LABELS = {'1':'batFish','2':'clownFish','3':'crab','4':'crocodile','5':'crocodileFish','6':'fish','7':'flounder','8':'frogFish','9':'ghostPipefish','10':'leafySeaDragon','11':'octopus','12':'pagurian','13':'pipefish','14':'scorpionFish','15':'seaHorse','16':'shrimp','17':'slug','18':'starFish','19':'stingaree','20':'turtle','21':'ant','22':'bug','23':'cat','24':'caterpillar','25':'centipede','26':'chameleon','27':'cheetah','28':'deer','29':'dog','30':'duck','31':'gecko','32':'giraffe','33':'grouse','34':'human','35':'kangaroo','36':'leopard','37':'lion','38':'lizard','39':'monkey','40':'rabbit','41':'reccoon','42':'sciuridae','43':'sheep','44':'snake','45':'spider','46':'stickInsect','47':'tiger','48':'wolf','49':'worm','50':'bat','51':'bee','52':'beetle','53':'bird','54':'bittern','55':'butterfly','56':'cicada','57':'dragonfly','58':'frogmouth','59':'grasshopper','60':'heron','61':'katydid','62':'mantis','63':'mockingbird','64':'moth','65':'owl','66':'owlfly','67':'frog','68':'toad','69':'other'}

NUM_CLASSES = 69

cod_training_root = os.path.join(datasets_root, 'train/') 
chameleon_path = os.path.join(datasets_root, 'test/CHAMELEON')
camo_path = os.path.join(datasets_root, 'test/CAMO')
cod10k_path = os.path.join(datasets_root, 'test/COD10K')
nc4k_path = os.path.join(datasets_root, 'test/NC4K')


