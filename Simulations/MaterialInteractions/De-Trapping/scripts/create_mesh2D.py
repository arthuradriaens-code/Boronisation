'''
Created on Jan 16, 2021

@author: Stephen
'''
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri
import scipy.spatial
import random
import itertools
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

number_of_decimals = "11"
np.set_printoptions(formatter={'float': lambda x: "{0:0." + number_of_decimals + "f}".format(x)+"0"})


class Mesh():
    """
    This is the class that contains all the stuff to create a mesh for RustBCA. currently the only default shapes are N-point polygons and random points within the simulation boundaries
    """
    def __init__(self, length_unit, xmax, xmin, ymax, ymin, energy_barrier_thickness=1.7645653881793786e-4):
        """
        returns none.
        The init functions requires the length units, x limits and y limits of the simulation.
        The x and y limits are assumed to be on the points of a square.
        """
        if xmin >= xmax:
            print("xmin should be less than xmax")
            assert xmin < xmax
        if ymin >= ymax:
            print("ymin should be less than ymax")
            assert ymin < ymax
        
        self.trianglelist = []
        
        self.Simulation_boundaries = [[xmax, ymax], [xmax, ymin], [xmin, ymin], [xmin, ymax]]
        self.points = []
        self.shapes = []
        self.xmax, self.xmin, self.ymax, self.ymin = xmax, xmin, ymax,ymin
        if not type(length_unit) == str:
            print("length_units should be a string")
            assert type(length_unit) == str
        self.length_unit = length_unit
        self.electronic_stopping_corrections = []
        self.energy_barrier_thickness = energy_barrier_thickness

    
    def N_gon(self, radius, n_points, number_densities, x_offset = 0.0, y_offset = 0.0, theta_offset = 0.0):
        """
        returns true on completion
        This creates a polygon with n-points.
        The x and y offset determine the center point
        theta_offset is in radians
        
        easy shape examples:
        circle: N_gon(5,100,1)
        square: N_gon(2, 4, 2)
        offset square: N_gon(5, 4, 2, 1, 1, np.pi/4)  
        """
        n_points = int(n_points)
        
        n_points -= 0
        dtheta = np.pi*2/n_points
        
        temp_points = []
        for i in range(n_points):
            temp_points.append(
                Point(x_offset + radius*math.cos(dtheta*i + theta_offset), y_offset + radius*math.sin(dtheta*i + theta_offset))
                )

        poly = Polygon(temp_points)
        
        center = Point(x_offset,y_offset)
        if center.within(poly):
            temp_points.append(center)

        self.points += temp_points
        
        self.shapes.append([poly, number_densities])
        
        return True
    
    def triangle(self, arm1, arm2, theta, number_densities, x_offset = 0, y_offset = 0, theta_offset = 0):
        '''
        returns True on completion
        creates a single triangle with arm lengths arm2 and arm2 and major angle theta (radians)
        The x and y offset determine the center point
        theta_offset determines the rotation of the triangle
        '''
        point1 = Point(x_offset + arm1*math.cos(theta_offset), y_offset + arm1*math.sin(theta_offset))
        point2 = Point(x_offset + arm2*math.cos(theta + theta_offset), y_offset + arm2*math.sin(theta + theta_offset))
        center_point = Point(x_offset, y_offset)
        
        temp_points = [point1,point2,center_point]
        
        self.points += temp_points
        self.shapes.append([Polygon(temp_points), number_densities])
        
        
    def rectangle(self, length_x, length_y, number_densities, x_offset = 0.0, y_offset = 0.0, theta_offset = 0.0):
        '''
        returns True on completion
        creates a single rectangle with center (x_offset, y_offset) and side lengths length1 and length2
        length1 is the x-axis
        theta_offset rotates around the center point (radians) <- NOT IMPLEMENTED CORRECTLY CURRENTLY
        '''
        temp_points = []
        for i in range(0,2):
            for j in range(0,2):
                temp_points.append(
                    Point(x_offset + (-1)**i * length_x/2.0, y_offset + (-1)**(i ^ j) * length_y/2.0)
                    )
        
        #print(temp_points)
        
        
        poly = Polygon(temp_points)
        
        center = Point(x_offset,y_offset)
        #print(center.within(poly))
        if center.within(poly):
            temp_points.append(center)
        

        self.points += temp_points
        self.shapes.append([poly, number_densities])
        
        return True
    
    def rectangle_grid(self, n_x, n_y, total_length_x, total_length_y, number_densities, bottom_left_x = 0.0, bottom_left_y = 0.0):
        '''
        returns True on completion
        Uses the total length and number of rectangles per side to calculate how rectangles should be made
        The anchor point is the bottom left most point of the grid of rectangles
        Since rectangles are defined the way they are in the code below it has to divide the number of rectangles per side by 2 to get what people intuit as the number of rectangles made of two triangles per side
        '''
        
        n_x, n_y = n_x, n_y 
        
        length_x = total_length_x/n_x
        length_y = total_length_y/n_y
        
        for j in range(n_y):
            for i in range(n_x):
                self.rectangle(length_x, length_y, number_densities, x_offset = (bottom_left_x + length_x/2 + i*length_x), y_offset = (bottom_left_y + length_y/2 + j*length_y))
        
        return True
    
    def add_Uniform_random(self, n_points):
        """
        returns True on completion.
        Adds uniformly random points to break up the triangles into fairly random triangles.
        **Note**
        Will cause some shapes to have parts of them be labeled the incorrect number density. 
        """
        temp_points = []
        for _ in range(n_points):
            temp_points.append(
                Point(random.uniform(self.xmin, self.xmax), random.uniform(self.ymin, self.ymax))
                )
        #self.shapes.append(Polygon(temp_points))
        self.points += temp_points
        return True
    
    def clean_points(self):
        temp = []
        for point in self.points:
            temp.append([point.x,point.y])
        
        temp.sort()
        temp = list(temp for temp,_ in itertools.groupby(temp))
        
        for i in range(len(temp)):
            temp[i] = Point(temp[i])
        
        self.points = temp
        
        return True
    
    def get_Points(self):
        """
        Deprecated - Don't use
        returns all of the individual points in the simulation as a np array
        """
        #self.clean_points()
        temp = []
        for point in self.points:
            temp.append([point.x,point.y])
        
        return np.asarray(temp)
    
    def generate_Triangles(self):
        """
        returns a scipy.spatial.Delauny object
        Generates triangles via the Delauny method:
        see scipy.spatial.Delaunay
        """
        #print(len(self.get_Points()))
        return scipy.spatial.Delaunay(self.get_Points())
    
    def return_Triangles(self):
        """
        returns 2 lists : points, material densities
        Creates and Correlates the triangles to their number densities.
        
        Will throw errors shapely errors if lines overlap
        """
        
        points = self.get_Points()
        tri = self.generate_Triangles()
        
        point_output = []
        triangles = []
        
        for i in range(len(tri.simplices)):
            point_output.append([points[tri.simplices[i,0], 0], points[tri.simplices[i,1], 0], points[tri.simplices[i,2], 0], points[tri.simplices[i,0], 1], points[tri.simplices[i,1], 1], points[tri.simplices[i,2], 1]])
            triangles.append(Polygon([points[tri.simplices[i,0]], points[tri.simplices[i,1]], points[tri.simplices[i,2]]]))
            
        #print("Length of tri.simplices " + str(len(tri.simplices)))
        #print("Length of triangles " + str(len(triangles)))
        
        temp_material_densities = [0.0]*(len(triangles))
        self.electronic_stopping_corrections = [1.0]*(len(triangles))
        #print("Len of triangles " + str(len(triangles)))
        for i, triangle in enumerate(triangles):
            for shape, number_densities in self.shapes:
                if shape.contains(triangle) or shape.relate_pattern(triangle, 'FF2FF1212') or shape.relate_pattern(triangle, '212101212'):
                    #print(shape.relate(triangle))
                    temp_material_densities[i] = list(number_densities)
                else:
                    #print(shape.relate(triangle))
                    continue
        #print(point_output)
        #print(len(temp_material_densities))
        return point_output, temp_material_densities
    
    def print_Triangles(self):
        """
        returns True on completion
        Use only as a check with number densities as integers from 0 to 3
        Plots the triangles with their colors based on their number densities
        Number densities should be integers currently
        """
        triangle_list, material_densities = self.return_Triangles()
    
        #print(len(triangle_list))
    
        #print(len(material_densities))
        
        self.color_dict = {0:"b", 1:"g", 2:"r", 3:"c", 4:"m"}
        
        triangles = []
        for t in triangle_list:
            triangles.append(
                matplotlib.tri.Triangulation(t[0:3], t[3:6])
                )
    
        #print(len(triangles))
        #print(type(triangles[0]))
        
        #plt.triplot(points[:,0], points[:,1], tri.simplices)
        plt.xlim(self.Simulation_boundaries[-1][0], self.Simulation_boundaries[1][0])
        plt.ylim(self.Simulation_boundaries[1][1], self.Simulation_boundaries[0][1])
        for i, triangle in enumerate(triangles):
            if type(material_densities[i]) == list:
                plt.triplot(triangle, self.color_dict[material_densities[i][0]]+ "-", )
            else:
                plt.triplot(triangle, self.color_dict[material_densities[i]]+ "-", )
        plt.show()
        return True
    
    def write_to_file(self, dump_to_file = False):
        '''
        returns a dictionary.
        Turn the required mesh2D stuff into a format that TOML and RustBCA like. 
        Optional to write to a .toml file instead of returning a dictionary
        '''
        triangle_list, material_densities = self.return_Triangles()
        Simulation_Boundaries = self.Simulation_boundaries
        
        material_boundary_points = []
        for shape, _ in self.shapes:
            material_boundary_points += shape.exterior.coords
        for i, point in reversed(list(enumerate(material_boundary_points))):
            for shape, _ in self.shapes:
                pointPoint = Point(point)
                if pointPoint.within(shape):
                    material_boundary_points.pop(i)
        

        material_boundary_points.sort()
        material_boundary_points = list(material_boundary_points for material_boundary_points,_ in itertools.groupby(material_boundary_points))
        #print(len(material_boundary_points))
        
        electronic_stopping_correction_factors = self.electronic_stopping_corrections
        
        '''import decimal
        decimal.getcontext().prec = int(number_of_decimals) + 2
        def have_ending_zeros(lis):
            #lis = np.asarray(lis)
            
            if type(lis[0]) == tuple:
                for i in range(len(lis)):
                    lis[i] = list(lis[i])
            
            for i in range(len(lis)):
                if type(lis[i]) == np.float64 or type(lis[i]) == np.float32 or type(lis[i]) == int or type(lis[i]) == float:
                    lis[i] =  decimal.Decimal(("{0:0." + str(len(str(lis[i]))) + "f}").format(lis[i])+"0")#.quantize(Decimal("1." + "0"*(int(number_of_decimals))))
                    #lis[i] =  decimal.Decimal(lis[i]).quantize(Decimal("1." + "0"*(int(number_of_decimals))))
                else:
                    if type(lis[i]) == type(decimal.Decimal(1.0)):
                                pass
                    for j in range(len(lis[i])):
                        if type(lis[i][j]) == np.float64 or type(lis[i][j]) == np.float32 or type(lis[i][j]) == int or type(lis[i][j]) == float:
                            lis[i][j] =  decimal.Decimal(("{0:0." + str(len(str(lis[i][j]))) + "f}").format(lis[i][j])+"0")#.quantize(Decimal("1." + "0"*(int(number_of_decimals))))
                            #lis[i][j] =  decimal.Decimal(lis[i][j]).quantize(Decimal("1." + "0"*(int(number_of_decimals))))
                        else:
                            if type(lis[i][j]) == type(decimal.Decimal(1.0)):
                                pass
                            for k in range(len(lis[i][j])):
                                if type(lis[i][j][k]) == np.float64 or type(lis[i][j][k]) == np.float32 or type(lis[i][j][k]) == int or type(lis[i][j][k]) == float:
                                    lis[i][j][k] =  decimal.Decimal(("{0:0." + str(len(str(lis[i][j][k]))) + "f}").format(lis[i][j][k])+"0")#.quantize(Decimal("1." + "0"*(int(number_of_decimals))))
                                    #lis[i][j][k] =  decimal.Decimal(lis[i][j][k]).quantize(Decimal("1." + "0"*(int(number_of_decimals))))
                                else:
                                    pass

            return lis
        '''
        
        file = open("Mesh2D.toml", "w")
        file.seek(0)
        
        '''
        temp_dict = {
            "mesh_2d_input" : {
                "length_unit":self.length_unit,
                "energy_barrier_thickness": decimal.Decimal(("{0:0." + str(len(str(self.energy_barrier_thickness))) + "f}").format(self.energy_barrier_thickness)+"0"),
                "triangles": have_ending_zeros(triangle_list),
                "densities":have_ending_zeros(material_densities),
                "material_boundary_points": have_ending_zeros(material_boundary_points),
                "simulation_boundary_points":have_ending_zeros(Simulation_Boundaries),
                "electronic_stopping_correction_factors":have_ending_zeros(electronic_stopping_correction_factors)
            }
        }'''
        temp_dict = {
            "mesh_2d_input" : {
                "length_unit":self.length_unit,
                "energy_barrier_thickness":self.energy_barrier_thickness,
                "triangles": triangle_list,
                "densities": material_densities,
                "material_boundary_points": material_boundary_points,
                "simulation_boundary_points": Simulation_Boundaries,
                "electronic_stopping_correction_factors": electronic_stopping_correction_factors
            }
        }
        if dump_to_file:
            import toml
            toml.dump(temp_dict, file, encoder=toml.TomlNumpyEncoder())
        
        #print(type(triangle_list))
        
        return temp_dict
if __name__ == "__main__":
    
    import timeit
    start  = timeit.default_timer()
    mesh = Mesh("MICRON", 50.1, -0.1, 50.1, -0.1)
    
    #mesh.N_gon(2, 4, [1], 1, 1, -np.pi/4 )
    #mesh.rectangle(1, 1, [2], 1, 1)
    mesh.rectangle_grid(10, 10, 50, 50, [5.305e11]) #Actual Number = 5.305e11
    
    #mesh.print_Triangles()
    mesh.write_to_file(True)
    
    
    
    end = timeit.default_timer()
    
    triangle_list, material_densities = mesh.return_Triangles()
    
    print(str(end-start) + "s to do " + str(len(triangle_list)) + " triangles" )
    
    #code to plot out the example triangles for boron_nitride
    '''
    triangles = [ [ 0.0, 0.01, 0.0, 0.5, 0.5, -0.5,], [ 0.0, 0.01, 0.01, -0.5, 0.5, -0.5,], [ 0.01, 0.01, 0.04, -0.5, 0.5, -0.5,], [ 0.01, 0.04, 0.04, 0.5, 0.5, -0.5,], [ 0.04, 0.5, 0.04, 0.5, 0.5, -0.5,], [ 0.04, 0.5, 0.5, -0.5, 0.5, -0.5,],]
    triangles = np.asarray(triangles)
    print(np.shape(triangles))
    
    points = []
    
    for i in range(len(triangles)):
        points.append([triangles[i,0], triangles[i,3]])
        points.append([triangles[i,1], triangles[i,4]])
        points.append([triangles[i,2], triangles[i,5]])
    
    
    import itertools
    points.sort()
    points = list(points for points,_ in itertools.groupby(points))
    
    points = np.asarray(points)
    print(np.shape(points))
    plt.triplot(points[:,0], points[:,1])
    plt.show()
    '''