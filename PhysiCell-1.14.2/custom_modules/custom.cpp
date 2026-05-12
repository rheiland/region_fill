/*
###############################################################################
# If you use PhysiCell in your project, please cite PhysiCell and the version #
# number, such as below:                                                      #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1].    #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# See VERSION.txt or call get_PhysiCell_version() to get the current version  #
#     x.y.z. Call display_citations() to get detailed information on all cite-#
#     able software used in your PhysiCell application.                       #
#                                                                             #
# Because PhysiCell extensively uses BioFVM, we suggest you also cite BioFVM  #
#     as below:                                                               #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1],    #
# with BioFVM [2] to solve the transport equations.                           #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# [2] A Ghaffarizadeh, SH Friedman, and P Macklin, BioFVM: an efficient para- #
#     llelized diffusive transport solver for 3-D biological simulations,     #
#     Bioinformatics 32(8): 1256-8, 2016. DOI: 10.1093/bioinformatics/btv730  #
#                                                                             #
###############################################################################
#                                                                             #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)     #
#                                                                             #
# Copyright (c) 2015-2021, Paul Macklin and the PhysiCell Project             #
# All rights reserved.                                                        #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
# this list of conditions and the following disclaimer.                       #
#                                                                             #
# 2. Redistributions in binary form must reproduce the above copyright        #
# notice, this list of conditions and the following disclaimer in the         #
# documentation and/or other materials provided with the distribution.        #
#                                                                             #
# 3. Neither the name of the copyright holder nor the names of its            #
# contributors may be used to endorse or promote products derived from this   #
# software without specific prior written permission.                         #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  #
# POSSIBILITY OF SUCH DAMAGE.                                                 #
#                                                                             #
###############################################################################
*/

#include "./custom.h"
#include <fstream>
#include <sstream>
#include <array>

std::vector<std::vector<std::array<double, 2>>> polygons;

bool point_in_polygon(double px, double py, const std::vector<std::array<double, 2>>& poly);
                      
void create_cell_types( void )
{
	// set the random seed 
	if (parameters.ints.find_index("random_seed") != -1)
	{
		SeedRandom(parameters.ints("random_seed"));
	}
	
	/* 
	   Put any modifications to default cell definition here if you 
	   want to have "inherited" by other cell types. 
	   
	   This is a good place to set default functions. 
	*/ 
	
	initialize_default_cell_definition(); 
	cell_defaults.phenotype.secretion.sync_to_microenvironment( &microenvironment ); 
	
	cell_defaults.functions.volume_update_function = standard_volume_update_function;
	cell_defaults.functions.update_velocity = standard_update_cell_velocity;

	cell_defaults.functions.update_migration_bias = NULL; 
	cell_defaults.functions.update_phenotype = NULL; // update_cell_and_death_parameters_O2_based; 
	cell_defaults.functions.custom_cell_rule = NULL; 
	cell_defaults.functions.contact_function = NULL; 
	
	cell_defaults.functions.add_cell_basement_membrane_interactions = NULL; 
	cell_defaults.functions.calculate_distance_to_membrane = NULL; 
	
	/*
	   This parses the cell definitions in the XML config file. 
	*/
	
	initialize_cell_definitions_from_pugixml(); 

	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/
		
	build_cell_definitions_maps(); 

	/*
	   This intializes cell signal and response dictionaries 
	*/

	setup_signal_behavior_dictionaries(); 	

	/*
       Cell rule definitions 
	*/

	setup_cell_rules(); 

	/* 
	   Put any modifications to individual cell definitions here. 
	   
	   This is a good place to set custom functions. 
	*/ 
	
    Cell_Definition* pCD = find_cell_definition( "barrier0"); 
	pCD->functions.custom_cell_rule = barrier_custom_cell_rule; 
	// cell_defaults.functions.update_phenotype = phenotype_function; 
	// cell_defaults.functions.custom_cell_rule = custom_function; 
	// cell_defaults.functions.contact_function = contact_function; 
	
	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/
		
	display_cell_definitions( std::cout ); 
	
	return; 
}

void setup_microenvironment( void )
{
	// set domain parameters 
	
	// put any custom code to set non-homogeneous initial conditions or 
	// extra Dirichlet nodes here. 
	
	// initialize BioFVM 
	
	initialize_microenvironment(); 	
	
	return; 
}

void setup_tissue( void )
{
	double Xmin = microenvironment.mesh.bounding_box[0]; 
	double Ymin = microenvironment.mesh.bounding_box[1]; 
	double Zmin = microenvironment.mesh.bounding_box[2]; 

	double Xmax = microenvironment.mesh.bounding_box[3]; 
	double Ymax = microenvironment.mesh.bounding_box[4]; 
	double Zmax = microenvironment.mesh.bounding_box[5]; 
	
	if( default_microenvironment_options.simulate_2D == true )
	{
		Zmin = 0.0; 
		Zmax = 0.0; 
	}
	
	double Xrange = Xmax - Xmin; 
	double Yrange = Ymax - Ymin; 
	double Zrange = Zmax - Zmin; 
	
	// create some of each type of cell 
	
	Cell* pC;
	
	// for( int k=0; k < cell_definitions_by_index.size() ; k++ )
	// {
	// 	Cell_Definition* pCD = cell_definitions_by_index[k]; 
	// 	std::cout << "Placing cells of type " << pCD->name << " ... " << std::endl; 
	// 	for( int n = 0 ; n < parameters.ints("number_of_cells") ; n++ )
	// 	{
	// 		std::vector<double> position = {0,0,0}; 
	// 		position[0] = Xmin + UniformRandom()*Xrange; 
	// 		position[1] = Ymin + UniformRandom()*Yrange; 
	// 		position[2] = Zmin + UniformRandom()*Zrange; 
			
	// 		pC = create_cell( *pCD ); 
	// 		pC->assign_position( position );
	// 	}
	// }
	// std::cout << std::endl; 
	
	// read polygon vertices from one or more CSV files (x, y, z columns with header)
    // NOTE: this file path may not work on Windows
	std::vector<std::string> csv_files = {
		"config/heart_poly.csv",
		"config/blob_poly.csv"
	};

    // should check it exists
    Cell_Definition* pCD = cell_definitions_by_name["barrier0"];

    std::cout << "----- list of all cell defns (names):\n";
    for (auto& cd_name: PhysiCell::cell_definitions_by_name) 
    {
        std::cout << cd_name.second << std::endl;
    }

	for (const auto& csv_file : csv_files)
	{
		std::ifstream file(csv_file);
		if (!file.is_open())
		{
			std::cerr << "Error: could not open " << csv_file << std::endl;
			continue;
		}
		std::vector<std::array<double, 2>> poly;
		std::string line;
		std::getline(file, line); // skip header
		while (std::getline(file, line))
		{
			std::istringstream ss(line);
			std::string token;
			std::array<double, 2> pt;
			std::getline(ss, token, ','); pt[0] = std::stod(token);
			std::getline(ss, token, ','); pt[1] = std::stod(token);
	        // std::cout << '  parameters.bools.find_index("cells_on_polygon") = ' << parameters.bools.find_index("cells_on_polygon") << std::endl;
	        // if ( parameters.bools.find_index("cells_on_polygon") == true)
            // {
            //     pC = create_cell( *pCD ); 
	        //     pC->assign_position( pt[0], pt[1], 0.0 );
            // }
			poly.push_back(pt);
		}
		file.close();
		polygons.push_back(poly);
	}
    std::cout << "------ setup_tissue created # polygons= " << polygons.size() << std::endl;

	// load cells from your CSV file (if enabled)
	load_cells_from_pugixml();
	set_parameters_from_distributions();
	
	return; 
}

std::vector<std::string> my_coloring_function( Cell* pCell )
{ return paint_by_number_cell_coloring(pCell); }

void custom_division_function( Cell* pCell1, Cell* pCell2 )
{
    pCell2->custom_data["parent_ID"] = pCell1->custom_data["parent_ID"];
}

void barrier_custom_cell_rule( Cell* pCell, Phenotype& phenotype, double dt )
{ 
    // std::cout << __FUNCTION__ << ": ID= "<< pCell->ID << ": x,y= " << pCell->position[0] << ", " << pCell->position[1] << std::endl;

    pCell->custom_data["my_ID"] = pCell->ID;

    int parent_ID = pCell->custom_data["parent_ID"];

    if ( point_in_polygon(pCell->position[0], pCell->position[1], polygons[parent_ID]) )
        pCell->custom_data["inside"] = 1;
    else
        pCell->custom_data["inside"] = 0;

    return;
} 

// -------------  utility fns ------------------
bool point_in_polygon(double px, double py,
                      const std::vector<std::array<double, 2>>& poly)
{
    int n = static_cast<int>(poly.size());
    // if (n < 40)
    // {
    //     std::cout << __FUNCTION__ << ":  n= " << n << std::endl;

    // }

    bool inside = false;
    int j = n - 1;

    for (int i = 0; i < n; ++i) {
        double iy = poly[i][1], ix = poly[i][0];
        double jy = poly[j][1], jx = poly[j][0];

        bool straddles = (iy > py) != (jy > py);
        if (straddles) {
            double x_intersect = (jx - ix) * (py - iy) / (jy - iy) + ix;
            if (px < x_intersect)
                inside = !inside;
        }
        j = i;
    }
    // if (inside) std::cout << "   -- inside is True!\n";
    return inside;
}