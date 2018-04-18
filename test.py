import numpy as np
import math
import matplotlib.pyplot as plt
import time

grid_resolution = 100 #stevilo delov na katerega razdeliš vsako dimenzijo prostora risanja

interaction_map = np.zeros((grid_resolution,grid_resolution)) #slika interakcije

dimension = 0.001 #dimenzija celotnega prostora risanja


dr = dimension/grid_resolution/2.1 #korak leta delca
no_interaction_counter = 0 #stetje delcev ki so pobegnili iz obmocja risanje in niso reagirali

for i in range(0,10000):

    #generiraj nakljucni kot
    angle = 2*math.pi*np.random.uniform(0,1)

    #belezi v kateri celici si ze preizkusil interakcijo
    visited_cells = np.zeros((grid_resolution, grid_resolution))


    #polozaj delca
    r = dr
    x_coordinate = r * math.cos(angle)
    y_coordinate = r * math.sin(angle)
    interaction_cell = []


    while math.fabs(y_coordinate) < dimension/2 and math.fabs(x_coordinate) < dimension/2:

        #iz polazaja delca v kordinatni sistem matrike
        x_grid = math.floor(x_coordinate) + int(grid_resolution/2)
        y_grid = math.floor(y_coordinate) + int(grid_resolution/2)

        #novi polozaj delca
        r = r + dr
        x_coordinate = r * math.cos(angle)
        y_coordinate = r * math.sin(angle)

        #verjetnost interakcije
        interaction_propability = math.exp(-r)*np.random.uniform(0, 1)

        #je delac že bil v tej celici
        if visited_cells[x_grid,y_grid] == 0 and interaction_propability > 0.5:
            interaction_cell = [x_grid,y_grid]
            break

        visited_cells[x_grid, y_grid] = 1
    #dodaj interakcijo v mesto v matriko oziroma preistej pobegli delec
    if interaction_cell == []:
        no_interaction_counter = no_interaction_counter +1
    else:
        interaction_map[interaction_cell[0],interaction_cell[1]] = interaction_map[interaction_cell[0],interaction_cell[1]] + 1

# izpis pobeglih
print("escaped: " + str(no_interaction_counter))

#risanje interakcij
plt.subplot(2,1,1)

plt.imshow(interaction_map,cmap=plt.get_cmap('YlOrRd'))
plt.colorbar()

#presek polja interakcij
plt.subplot(2,1,2)
plt.plot(interaction_map[int(grid_resolution/2),:])
plt.show()

