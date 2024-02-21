# Jupyter-notebook-For-Mapping-Genetics-data
This is a code used to buillt a map using a Genetic data analysis from Structure analysis. 
This are the packages I used to 
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
# Assuming you have a GeoDataFrame 'shp' that contains country borders
shp = gpd.read_file('C:/Users/Raster/continentsshp06.shp')


# Set the path to your CSV file
csv_path = r"C:\Users\FM_K.csv"
# Load data into a DataFrame
final = pd.read_csv(csv_path)

# Display the first few rows of the DataFrame
print(final.head())
shapefile_path = r"C:\Users\Raster\continentsshp06.shp"
countries = gpd.read_file(shapefile_path)
final['Latitude'] = pd.to_numeric(final['Latitude'])
final['Longitude'] = pd.to_numeric(final['Longitude'])
# Adjust the smooth parameter as needed
smooth_parameter = 1e-3
margin = 6
# Create a grid for plotting with a margin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from scipy.interpolate import Rbf
from sklearn.linear_model import Ridge
grid_x, grid_y = np.meshgrid(
    np.arange(final['Longitude'].min() - margin, final['Longitude'].max() + margin, 0.25),
    np.arange(final['Latitude'].min() - margin, final['Latitude'].max() + margin, 0.25)
)
# Create the Rbf interpolation function with regularization
int1 = Rbf(final['Longitude'], final['Latitude'], final['K3pop1'], function='linear', smooth=smooth_parameter)
int1 = Rbf(final['Longitude'], final['Latitude'], final['K3pop2'], function='linear', smooth=smooth_parameter)
int1 = Rbf(final['Longitude'], final['Latitude'], final['K3pop3'], function='linear', smooth=smooth_parameter)
# Interpolate values on the grid
grid_z = int1(grid_x, grid_y)
import geopandas as gpd
import matplotlib.pyplot as plt

# Plot the map
plt.figure(figsize=(20, 20))
cmap = 'RdBu' 
 # Choose a diverging colormap
plt.pcolormesh(grid_x, grid_y, grid_z, shading='auto', cmap=cmap, vmin=grid_z.min(), vmax=grid_z.max())
plt.colorbar(label='K=3')
shp = shp = gpd.read_file(r'C:\Users\Raster\continentsshp06.shp')
plt.xlim([final['Longitude'].min() - margin, final['Longitude'].max() + margin])
plt.ylim([final['Latitude'].min() - margin, final['Latitude'].max() + margin])
plt.scatter(final['Longitude'], final['Latitude'], c='black', marker='.', s=3)
shp.boundary.plot(ax=plt.gca(), linewidth=1, color='black')
# Save the plot
output_path_tiff = r'C:\Users\\K=3_figure.tif'

plt.savefig(output_path_tiff, format='tiff', dpi=300)

plt.show()
