#data-----------
from simple_image_download import simple_image_download as sim

my_downloader = sim.Downloader()

my_downloader.directory = 'my_dir/'
# Change File extension type
my_downloader.extensions = '.jpg'
print(my_downloader.extensions)

#my_downloader.download('Rollceroyce', limit=100)
#my_downloader.download('bmw', limit=100)
my_downloader.download('honda', limit=100)
my_downloader.download('toyota', limit=100)