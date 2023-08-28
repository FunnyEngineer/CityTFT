from bs4 import BeautifulSoup
import pdb 

with open('data/SRLOD3.1_Annual_results.xml', 'r') as f:
    data = f.read()
 
# Passing the stored data inside
# the beautifulsoup parser, storing
# the returned object
Bs_data = BeautifulSoup(data, "xml")
pdb.set_trace()

# Finding all instances of tag
# `unique`
b_unique = Bs_data.find_all('unique')
 
print(b_unique)
 
# Using find() to extract attributes
# of the first instance of the tag
b_name = Bs_data.find('child', {'name':'Frank'})
 
print(b_name)
 
# Extracting the data stored in a
# specific attribute of the
# `child` tag
value = b_name.get('test')
 
print(value)