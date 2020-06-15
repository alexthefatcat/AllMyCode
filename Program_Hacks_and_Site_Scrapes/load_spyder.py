 
import tarfile, pickle, io

def tar_file_to_bytes(filename):
    tar_dict = {}
    tar = tarfile.open(filename, "r")
    tarnames = list(tar.getnames())
    for member in tarnames :
        tar_dict[member] = tar.extractfile(member).read()
    v = tar_dict[ tarnames[0] ]
    return v
 
# use this
def read_spydata(filename):
    v = tar_file_to_bytes(filename)
    data = pickle.loads(v)   
    return data



if __name__ == "__main__":
    
    filename = r"C:\Users\Alexm\Desktop\example_.spydata"
    data = read_spydata(filename)










if False:
    def __old_requires_intermediate_file_creation(filename):
        f_intermeidate = "date_inter.pickle"
        v = tar_file_to_bytes(filename)
        #   2) save the bytes of the untar
        with open(f_intermeidate , 'wb') as file:
            file.write(v)
        data2 = pickle.load(io.BytesIO(v))    
        #   3) load and unpickle in variable called data 
        with open(f_intermeidate , "rb") as f:
            data = pickle.load(f)
        return data,data2
        
        
 
        
        
        



