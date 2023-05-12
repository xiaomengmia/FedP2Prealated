cd ../
sudo rm /opt/warwick/Tmp/*
sudo rm /opt/warwick/Project/*
sudo rm /opt/warwick/ModelFile/*th
sudo rm /opt/warwick/Log/*
sudo tar --exclude='./warwick/config' --exclude='./warwick/DatasetFile' -czvf ./warwick.tar.gz ./warwick
for host in 'pochun@192.168.2.163' 'pochun@192.168.2.164' 'pochun@192.168.2.165' 'pochun@192.168.2.166' 'pi@192.168.2.171' 'pochun@192.168.2.167' 'pochun@192.168.2.211' 'pochun@192.168.2.212' 'pi@192.168.2.172' 'pi@192.168.2.173'
do
  echo $host
  scp ./warwick.tar.gz $host:~/
done
timestamp=$(date +%s)
sudo mv ./warwick.tar.gz ./warwick.$timestamp.tar.gz
cd ./warwick
