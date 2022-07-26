#############################################################################################
# Virtual Environment (Sanal ortam) ve Package Managemen (Paket Yönetimi)
#############################################################################################

# Sanal ortamların listelenmesi:
# conda env list

# Sanal ortam oluşturma:
# conda create -n myevn

# Sanal ortamı aktif etme:
# conda activate myevn

# Yüklü paketlerin listelenmesi:
# conda list

# Paket yükleme:
# conda install numpy

# Paket silme:
# conda remove package_name

# Belirli bir versiyona göre paket yükleme:
# conda install numpy=1.20.1

# Paket yükseltme:
# conda upgrade numpy

# tüm paketlerin yükseltilmesi:
# conda upgrade -all

# pip : (pypi) Paket Yönetim Aracı

# Paket yükleme
# pip install package_name

# Paket yükleme versiyona göre
# pip install package_name== 1.20.1

# var olan bir environment ' ı oluşturma
# conda env export > environment.yaml


students = ["John", "Mark", "Venessa", "Mariam"]

students_no = ["John", "Venessa"]

[student.lower() if student in students_no else student.upper() for student in students]

dictionary = {'a': 1,
              'b': 2,
              'c': 3,
              'd': 4}

dictionary.keys()
dictionary.values()
dictionary.items()

{i**2 for i in dictionary.values()}