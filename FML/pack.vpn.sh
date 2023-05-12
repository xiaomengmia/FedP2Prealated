cd ../
sudo rm /opt/warwick/Tmp/*
sudo rm /opt/warwick/Project/*
sudo rm /opt/warwick/ModelFile/*th
sudo rm /opt/warwick/Log/*
sudo tar --exclude='./warwick/config' --exclude='./warwick/DatasetFile' -czvf ./warwick.tar.gz ./warwick
for host in 'pochun@10.147.17.101' 'pochun@10.147.17.102' 'pochun@10.147.17.103' 'pochun@10.147.17.104' 'pi@10.147.17.201' 'pochun@10.147.17.105' 'pochun@10.147.17.106' 'pochun@10.147.17.107' 'pi@10.147.17.202' 'pi@10.147.17.203' 'pi@10.147.17.201' 
do
  echo $host
  scp ./warwick.tar.gz $host:~/
done
timestamp=$(date +%s)
sudo mv ./warwick.tar.gz ./warwick.$timestamp.tar.gz
cd ./warwick
