import os
import sys
from convert import Convert

if __name__ == "__main__":
  target_dir = 'graph'
  if os.path.exists(target_dir):
    print(target_dir, 'exists!')
    exit()
  os.mkdir(target_dir)

  c = Convert()
  c.convert('raw/user_user.dat', 'user', 'user', 'friend')
  c.convert('raw/user_business.dat', 'user', 'business', 'bought')
  c.convert('raw/user_compliment.dat', 'user', 'compliment', 'level')
  c.convert('raw/business_category.dat', 'business', 'category', 'belong')
  c.convert('raw/business_city.dat', 'business', 'city', 'locate')
  c.transfer(target_dir)
  c.split2(target_dir, 'user_business_bought.txt')
