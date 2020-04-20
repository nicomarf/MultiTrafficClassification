                          SCRIPT DE CRIACAO DE UM DATASET UNICO, AGRUPANDO DATASET ALEMAO, BELGA e CROATA



------------Deletar arquivos do alemao
rm -r AlemaoBelgaCroata/Alemao/Training
rm -r AlemaoBelgaCroata/Alemao/Testing

--------------Deletar belga ----------------
rm -r AlemaoBelgaCroata/Belga/Training
rm -r AlemaoBelgaCroata/Belga/Testing

-------------Deletar croata---------
rm -r AlemaoBelgaCroata/Croata/Training
rm -r AlemaoBelgaCroata/Croata/Testing

------------Copiar alemao , belga e croata

cp -R AlemaoTraf/Training AlemaoBelgaCroata/Alemao


cp -R BelgaTraf/Training AlemaoBelgaCroata/Belga
cp -R BelgaTraf/Testing AlemaoBelgaCroata/Belga

cp -R CroataTraf/Training AlemaoBelgaCroata/Croata
cp -R CroataTraf/Testing AlemaoBelgaCroata/Croata



---RENOMEAR TODOS ARQUIVOS COLOCANDO uma letra antes. b(belga) c(croacia)

-----------------TREINO-------------------------------------------

cd AlemaoBelgaCroata/Belga/Training
for diretorio in `ls`  ; do	
  cd ${diretorio}
  for sid in `ls  *.ppm`  ; do
      `mv "${sid}" b"${sid}"`
  done
  cd ..
done


cd AlemaoBelgaCroata/Croata/Training
for diretorio in `ls`  ; do	
  cd ${diretorio}
  for sid in `ls  *.ppm`  ; do
      `mv "${sid}" c"${sid}"`
  done
  cd ..
done

-------------------TESTE--------------------------- 
cd AlemaoBelgaCroata/Belga/Testing
for diretorio in `ls`  ; do	
  cd ${diretorio}
  for sid in `ls  *.ppm`  ; do
      `mv "${sid}" b"${sid}"`
  done
  cd ..
done


cd AlemaoBelgaCroata/Croata/Testing
for diretorio in `ls`  ; do	
  cd ${diretorio}
  for sid in `ls  *.ppm`  ; do
      `mv "${sid}" c"${sid}"`
  done
  cd ..
done
--------------------------CRIAR DIRETORIOS TRAINING ALEMAO--
cd AlemaoBelgaCroata/Alemao/Training

mkdir                   00043 00044 00045 00046 00047 00048 00049
mkdir 00050 00051 00052 00053 00054 00055 00056 00057 00058 00059
mkdir 00060 00061 00062 00063 00064 00065 00066 00067 00068 00069
mkdir 00070 00071 00072 00073 00074 00075 00076 00077 00078 00079
mkdir 00080 00081 00082 00083 00084 00085 00086 00087 
-----------------------COPIAR- TRAINING-----------------------



cd AlemaoBelgaCroata/Belga/Training
mv 00031/*.ppm AlemaoBelgaCroata/Alemao/Training/00009
mv 00017/*.ppm AlemaoBelgaCroata/Alemao/Training/00011
mv 00061/*.ppm AlemaoBelgaCroata/Alemao/Training/00012
mv 00019/*.ppm AlemaoBelgaCroata/Alemao/Training/00013
mv 00021/*.ppm AlemaoBelgaCroata/Alemao/Training/00014
mv 00028/*.ppm AlemaoBelgaCroata/Alemao/Training/00015
mv 00025/*.ppm AlemaoBelgaCroata/Alemao/Training/00016
mv 00022/*.ppm AlemaoBelgaCroata/Alemao/Training/00017
mv 00013/*.ppm AlemaoBelgaCroata/Alemao/Training/00018
mv 00003/*.ppm AlemaoBelgaCroata/Alemao/Training/00019
mv 00004/*.ppm AlemaoBelgaCroata/Alemao/Training/00020
mv 00005/*.ppm AlemaoBelgaCroata/Alemao/Training/00021
mv 00000/*.ppm AlemaoBelgaCroata/Alemao/Training/00022
mv 00002/*.ppm AlemaoBelgaCroata/Alemao/Training/00023
mv 00016/*.ppm AlemaoBelgaCroata/Alemao/Training/00024
mv 00010/*.ppm AlemaoBelgaCroata/Alemao/Training/00025
mv 00011/*.ppm AlemaoBelgaCroata/Alemao/Training/00026
mv 00007/*.ppm AlemaoBelgaCroata/Alemao/Training/00028
mv 00008/*.ppm AlemaoBelgaCroata/Alemao/Training/00029
mv 00034/*.ppm AlemaoBelgaCroata/Alemao/Training/00035
mv 00036/*.ppm AlemaoBelgaCroata/Alemao/Training/00036
mv 00037/*.ppm AlemaoBelgaCroata/Alemao/Training/00040




mv 00014/*.ppm AlemaoBelgaCroata/Alemao/Training/00043
mv 00038/*.ppm AlemaoBelgaCroata/Alemao/Training/00044
mv 00039/*.ppm AlemaoBelgaCroata/Alemao/Training/00045
mv 00006/*.ppm AlemaoBelgaCroata/Alemao/Training/00046
mv 00041/*.ppm AlemaoBelgaCroata/Alemao/Training/00047
mv 00045/*.ppm AlemaoBelgaCroata/Alemao/Training/00048
mv 00056/*.ppm AlemaoBelgaCroata/Alemao/Training/00049


mv 00001/*.ppm AlemaoBelgaCroata/Alemao/Training/00050
mv 00009/*.ppm AlemaoBelgaCroata/Alemao/Training/00051
mv 00012/*.ppm AlemaoBelgaCroata/Alemao/Training/00052

mv 00015/*.ppm AlemaoBelgaCroata/Alemao/Training/00053
mv 00018/*.ppm AlemaoBelgaCroata/Alemao/Training/00054
mv 00020/*.ppm AlemaoBelgaCroata/Alemao/Training/00055
mv 00023/*.ppm AlemaoBelgaCroata/Alemao/Training/00056
mv 00024/*.ppm AlemaoBelgaCroata/Alemao/Training/00057

mv 00026/*.ppm AlemaoBelgaCroata/Alemao/Training/00058
mv 00027/*.ppm AlemaoBelgaCroata/Alemao/Training/00059
mv 00029/*.ppm AlemaoBelgaCroata/Alemao/Training/00060
mv 00030/*.ppm AlemaoBelgaCroata/Alemao/Training/00061
mv 00032/*.ppm AlemaoBelgaCroata/Alemao/Training/00062
mv 00033/*.ppm AlemaoBelgaCroata/Alemao/Training/00063
mv 00035/*.ppm AlemaoBelgaCroata/Alemao/Training/00064

mv 00040/*.ppm AlemaoBelgaCroata/Alemao/Training/00065

mv 00042/*.ppm AlemaoBelgaCroata/Alemao/Training/00066
mv 00043/*.ppm AlemaoBelgaCroata/Alemao/Training/00067
mv 00044/*.ppm AlemaoBelgaCroata/Alemao/Training/00068
mv 00046/*.ppm AlemaoBelgaCroata/Alemao/Training/00069
mv 00047/*.ppm AlemaoBelgaCroata/Alemao/Training/00070
mv 00048/*.ppm AlemaoBelgaCroata/Alemao/Training/00071
mv 00049/*.ppm AlemaoBelgaCroata/Alemao/Training/00072
mv 00050/*.ppm AlemaoBelgaCroata/Alemao/Training/00073
mv 00051/*.ppm AlemaoBelgaCroata/Alemao/Training/00074
mv 00052/*.ppm AlemaoBelgaCroata/Alemao/Training/00075
mv 00053/*.ppm AlemaoBelgaCroata/Alemao/Training/00076
mv 00054/*.ppm AlemaoBelgaCroata/Alemao/Training/00077
mv 00055/*.ppm AlemaoBelgaCroata/Alemao/Training/00078  
mv 00057/*.ppm AlemaoBelgaCroata/Alemao/Training/00079
mv 00058/*.ppm AlemaoBelgaCroata/Alemao/Training/00080
mv 00059/*.ppm AlemaoBelgaCroata/Alemao/Training/00081
mv 00060/*.ppm AlemaoBelgaCroata/Alemao/Training/00082


cd AlemaoBelgaCroata/Croata/Training

mv 00017/*.ppm AlemaoBelgaCroata/Alemao/Training/00001
mv 00019/*.ppm AlemaoBelgaCroata/Alemao/Training/00002
mv 00020/*.ppm AlemaoBelgaCroata/Alemao/Training/00003
mv 00021/*.ppm AlemaoBelgaCroata/Alemao/Training/00004
mv 00022/*.ppm AlemaoBelgaCroata/Alemao/Training/00009
mv 00001/*.ppm AlemaoBelgaCroata/Alemao/Training/00011
mv 00016/*.ppm AlemaoBelgaCroata/Alemao/Training/00012
mv 00014/*.ppm AlemaoBelgaCroata/Alemao/Training/00013
mv 00015/*.ppm AlemaoBelgaCroata/Alemao/Training/00014
mv 00000/*.ppm AlemaoBelgaCroata/Alemao/Training/00018
mv 00004/*.ppm AlemaoBelgaCroata/Alemao/Training/00019
mv 00005/*.ppm AlemaoBelgaCroata/Alemao/Training/00020
mv 00006/*.ppm AlemaoBelgaCroata/Alemao/Training/00021
mv 00009/*.ppm AlemaoBelgaCroata/Alemao/Training/00023
mv 00010/*.ppm AlemaoBelgaCroata/Alemao/Training/00027
mv 00011/*.ppm AlemaoBelgaCroata/Alemao/Training/00028
mv 00012/*.ppm AlemaoBelgaCroata/Alemao/Training/00031
mv 00026/*.ppm AlemaoBelgaCroata/Alemao/Training/00038
mv 00027/*.ppm AlemaoBelgaCroata/Alemao/Training/00041


mv 00008/*.ppm AlemaoBelgaCroata/Alemao/Training/00043
mv 00024/*.ppm AlemaoBelgaCroata/Alemao/Training/00044
mv 00025/*.ppm AlemaoBelgaCroata/Alemao/Training/00045
mv 00007/*.ppm AlemaoBelgaCroata/Alemao/Training/00046
mv 00023/*.ppm AlemaoBelgaCroata/Alemao/Training/00047
mv 00029/*.ppm AlemaoBelgaCroata/Alemao/Training/00048
mv 00028/*.ppm AlemaoBelgaCroata/Alemao/Training/00049

mv 00002/*.ppm AlemaoBelgaCroata/Alemao/Training/00083
mv 00003/*.ppm AlemaoBelgaCroata/Alemao/Training/00084
mv 00013/*.ppm AlemaoBelgaCroata/Alemao/Training/00085
mv 00018/*.ppm AlemaoBelgaCroata/Alemao/Training/00086
mv 00030/*.ppm AlemaoBelgaCroata/Alemao/Training/00087

-----------------------------TESTING--------------------------
cd AlemaoBelgaCroata/Alemao
mkdir Testing
cd AlemaoBelgaCroata/Alemao/Testing
mkdir 00000 00001 00002 00003 00004 00005 00006 00007 00008 00009
mkdir 00010 00011 00012 00013 00014 00015 00016 00017 00018 00019
mkdir 00020 00021 00022 00023 00024 00025 00026 00027 00028 00029
mkdir 00030 00031 00032 00033 00034 00035 00036 00037 00038 00039
mkdir 00040 00041 00042 00043 00044 00045 00046 00047 00048 00049
mkdir 00050 00051 00052 00053 00054 00055 00056 00057 00058 00059
mkdir 00060 00061 00062 00063 00064 00065 00066 00067 00068 00069
mkdir 00070 00071 00072 00073 00074 00075 00076 00077 00078 00079
mkdir 00080 00081 00082 00083 00084 00085 00086 00087 

-----------------------COPIAR- Testing-----------------------



cd AlemaoBelgaCroata/Belga/Testing
mv 00031/*.ppm AlemaoBelgaCroata/Alemao/Testing/00009
mv 00017/*.ppm AlemaoBelgaCroata/Alemao/Testing/00011
mv 00061/*.ppm AlemaoBelgaCroata/Alemao/Testing/00012
mv 00019/*.ppm AlemaoBelgaCroata/Alemao/Testing/00013
mv 00021/*.ppm AlemaoBelgaCroata/Alemao/Testing/00014
mv 00028/*.ppm AlemaoBelgaCroata/Alemao/Testing/00015
mv 00025/*.ppm AlemaoBelgaCroata/Alemao/Testing/00016
mv 00022/*.ppm AlemaoBelgaCroata/Alemao/Testing/00017
mv 00013/*.ppm AlemaoBelgaCroata/Alemao/Testing/00018
mv 00003/*.ppm AlemaoBelgaCroata/Alemao/Testing/00019
mv 00004/*.ppm AlemaoBelgaCroata/Alemao/Testing/00020
mv 00005/*.ppm AlemaoBelgaCroata/Alemao/Testing/00021
mv 00000/*.ppm AlemaoBelgaCroata/Alemao/Testing/00022
mv 00002/*.ppm AlemaoBelgaCroata/Alemao/Testing/00023
mv 00016/*.ppm AlemaoBelgaCroata/Alemao/Testing/00024
mv 00010/*.ppm AlemaoBelgaCroata/Alemao/Testing/00025
mv 00011/*.ppm AlemaoBelgaCroata/Alemao/Testing/00026
mv 00007/*.ppm AlemaoBelgaCroata/Alemao/Testing/00028
mv 00008/*.ppm AlemaoBelgaCroata/Alemao/Testing/00029
mv 00034/*.ppm AlemaoBelgaCroata/Alemao/Testing/00035
mv 00036/*.ppm AlemaoBelgaCroata/Alemao/Testing/00036
mv 00037/*.ppm AlemaoBelgaCroata/Alemao/Testing/00040




mv 00014/*.ppm AlemaoBelgaCroata/Alemao/Testing/00043
mv 00038/*.ppm AlemaoBelgaCroata/Alemao/Testing/00044
mv 00039/*.ppm AlemaoBelgaCroata/Alemao/Testing/00045
mv 00006/*.ppm AlemaoBelgaCroata/Alemao/Testing/00046
mv 00041/*.ppm AlemaoBelgaCroata/Alemao/Testing/00047
mv 00045/*.ppm AlemaoBelgaCroata/Alemao/Testing/00048
mv 00056/*.ppm AlemaoBelgaCroata/Alemao/Testing/00049


mv 00001/*.ppm AlemaoBelgaCroata/Alemao/Testing/00050
mv 00009/*.ppm AlemaoBelgaCroata/Alemao/Testing/00051
mv 00012/*.ppm AlemaoBelgaCroata/Alemao/Testing/00052

mv 00015/*.ppm AlemaoBelgaCroata/Alemao/Testing/00053
mv 00018/*.ppm AlemaoBelgaCroata/Alemao/Testing/00054
mv 00020/*.ppm AlemaoBelgaCroata/Alemao/Testing/00055
mv 00023/*.ppm AlemaoBelgaCroata/Alemao/Testing/00056
mv 00024/*.ppm AlemaoBelgaCroata/Alemao/Testing/00057

mv 00026/*.ppm AlemaoBelgaCroata/Alemao/Testing/00058
mv 00027/*.ppm AlemaoBelgaCroata/Alemao/Testing/00059
mv 00029/*.ppm AlemaoBelgaCroata/Alemao/Testing/00060
mv 00030/*.ppm AlemaoBelgaCroata/Alemao/Testing/00061
mv 00032/*.ppm AlemaoBelgaCroata/Alemao/Testing/00062
mv 00033/*.ppm AlemaoBelgaCroata/Alemao/Testing/00063
mv 00035/*.ppm AlemaoBelgaCroata/Alemao/Testing/00064

mv 00040/*.ppm AlemaoBelgaCroata/Alemao/Testing/00065

mv 00042/*.ppm AlemaoBelgaCroata/Alemao/Testing/00066
mv 00043/*.ppm AlemaoBelgaCroata/Alemao/Testing/00067
mv 00044/*.ppm AlemaoBelgaCroata/Alemao/Testing/00068
mv 00046/*.ppm AlemaoBelgaCroata/Alemao/Testing/00069
mv 00047/*.ppm AlemaoBelgaCroata/Alemao/Testing/00070
mv 00048/*.ppm AlemaoBelgaCroata/Alemao/Testing/00071
mv 00049/*.ppm AlemaoBelgaCroata/Alemao/Testing/00072
mv 00050/*.ppm AlemaoBelgaCroata/Alemao/Testing/00073
mv 00051/*.ppm AlemaoBelgaCroata/Alemao/Testing/00074
mv 00052/*.ppm AlemaoBelgaCroata/Alemao/Testing/00075
mv 00053/*.ppm AlemaoBelgaCroata/Alemao/Testing/00076
mv 00054/*.ppm AlemaoBelgaCroata/Alemao/Testing/00077
mv 00055/*.ppm AlemaoBelgaCroata/Alemao/Testing/00078  
mv 00057/*.ppm AlemaoBelgaCroata/Alemao/Testing/00079
mv 00058/*.ppm AlemaoBelgaCroata/Alemao/Testing/00080
mv 00059/*.ppm AlemaoBelgaCroata/Alemao/Testing/00081
mv 00060/*.ppm AlemaoBelgaCroata/Alemao/Testing/00082


cd AlemaoBelgaCroata/Croata/Testing

mv 00017/*.ppm AlemaoBelgaCroata/Alemao/Testing/00001
mv 00019/*.ppm AlemaoBelgaCroata/Alemao/Testing/00002
mv 00020/*.ppm AlemaoBelgaCroata/Alemao/Testing/00003
mv 00021/*.ppm AlemaoBelgaCroata/Alemao/Testing/00004
mv 00022/*.ppm AlemaoBelgaCroata/Alemao/Testing/00009
mv 00001/*.ppm AlemaoBelgaCroata/Alemao/Testing/00011
mv 00016/*.ppm AlemaoBelgaCroata/Alemao/Testing/00012
mv 00014/*.ppm AlemaoBelgaCroata/Alemao/Testing/00013
mv 00015/*.ppm AlemaoBelgaCroata/Alemao/Testing/00014
mv 00000/*.ppm AlemaoBelgaCroata/Alemao/Testing/00018
mv 00004/*.ppm AlemaoBelgaCroata/Alemao/Testing/00019
mv 00005/*.ppm AlemaoBelgaCroata/Alemao/Testing/00020
mv 00006/*.ppm AlemaoBelgaCroata/Alemao/Testing/00021
mv 00009/*.ppm AlemaoBelgaCroata/Alemao/Testing/00023
mv 00010/*.ppm AlemaoBelgaCroata/Alemao/Testing/00027
mv 00011/*.ppm AlemaoBelgaCroata/Alemao/Testing/00028
mv 00012/*.ppm AlemaoBelgaCroata/Alemao/Testing/00031
mv 00026/*.ppm AlemaoBelgaCroata/Alemao/Testing/00038
mv 00027/*.ppm AlemaoBelgaCroata/Alemao/Testing/00041


mv 00008/*.ppm AlemaoBelgaCroata/Alemao/Testing/00043
mv 00024/*.ppm AlemaoBelgaCroata/Alemao/Testing/00044
mv 00025/*.ppm AlemaoBelgaCroata/Alemao/Testing/00045
mv 00007/*.ppm AlemaoBelgaCroata/Alemao/Testing/00046
mv 00023/*.ppm AlemaoBelgaCroata/Alemao/Testing/00047
mv 00029/*.ppm AlemaoBelgaCroata/Alemao/Testing/00048
mv 00028/*.ppm AlemaoBelgaCroata/Alemao/Testing/00049

mv 00002/*.ppm AlemaoBelgaCroata/Alemao/Testing/00083
mv 00003/*.ppm AlemaoBelgaCroata/Alemao/Testing/00084
mv 00013/*.ppm AlemaoBelgaCroata/Alemao/Testing/00085
mv 00018/*.ppm AlemaoBelgaCroata/Alemao/Testing/00086
mv 00030/*.ppm AlemaoBelgaCroata/Alemao/Testing/00087

------------------------------CRIAR csv do testing(belga,croata)------------
cd AlemaoBelgaCroata/Alemao
rm BC-final_testeBC.csv lixo1
find . -name '*.ppm' > saida1
grep 'Testing' saida1 > saida

for dire in `cat saida`  ; do
   echo ${dire} > lixo
   var=`cut lixo -d "/" -f 3`
   ptv=';'
   echo  ${dire}${ptv}${var} >> lixo1
done

sed -i '1i\Filename;ClassId\' lixo1

sed 's/;0000/;/' lixo1 > lixo2
sed 's/;000/;/' lixo2 > BC-final_testeBC.csv

grep '/b' BC-final_testeBC.csv > BelgaB.csv
sed -i '1i\Filename;ClassId\' BelgaB.csv
grep '/c' BC-final_testeBC.csv > CroataC.csv
sed -i '1i\Filename;ClassId\' CroataC.csv

sed 's/.\/Testing\///' BelgaB.csv > lixo
mv lixo BelgaB.csv

sed 's/.\/Testing\///' CroataC.csv > lixo
mv lixo CroataC.csv

rm lixo lixo1 lixo2 saida saida1

cp  AlemaoTraf/Testing/*.ppm AlemaoBelgaCroata/Alemao/Testing

-------------------------------FIM------------------------------------------------------




