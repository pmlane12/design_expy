import numpy as np
import pandas as pd
from patsy import ModelDesc


class Model(object):
	"""
	A class to represent the underlying world we are working with.

	Contains two class variables, both of which can be passed in as keyword arguments:

	population: The underlying data from which the experiment draws from. Can be:
				(1) a dataframe,
				(2) a dict with variable names as key and tuples with functions, args and kwargs as values,
				(3) a tuple or list containing one each of (1) and (2).

	potential_outcomes: A list of expressions as parsed by the pandas.eval method describing
						the relationships between variables, treatments and outcomes.
	"""


	def declare_population(self, population):
		"""
		Method that assigns the input passed as population argument to the class population variable.

		Is called when instantiating a Model object:
			model = Model(population=(df, dict))

		Or can be called by a Model instance to set the population:
			model = Model()
			model.declare_population((df, dict))
		"""

		if isinstance(population, (tuple,list)):
			if len(population) != 2:
				error = "Population argument is of length {}, was expecting length 2.".format(str(len(population)))
				raise ValueError(error)
			for population_input in population:
				if isinstance(population_input, pd.DataFrame):
					population_df = population_input.copy()
				elif isinstance(population_input, dict):
					population_dict = population_input
				else:
					input_types = ", ".join(type(x).__name__ for x in population)
					error = "Population argument collectiong is of types ({}), was expecting (Dataframe, dict)"
					raise TypeError(error.format(input_types))

			self.population = (population_df, population_dict)
		elif isinstance(population, pd.DataFrame):
			self.population = population.copy()
		elif isinstance(population, dict):
			self.population = population
		else:
			error = ("The population argument expects a dataframe, a dict, or a tuple or list of length 2"
					+" with one dataframe element and one dict element. Instead received an object of type {}.")
			raise TypeError(error.format(type(population).__name__))


	def declare_potential_outcomes(self, potential_outcomes):
		"""
		Method that assigns the input passed as potential_outcomes argument to the class potential_outcomes variable.

		Is called when instantiating a Model object:
			model = Model(potential_outcomes='X ~ Y')

		Or can be called by a Model instance to set the population:
			model = Model()
			model.declare_potential_outcomes('X ~ Y')
		"""

		if isinstance(potential_outcomes, str):
			self.potential_outcomes = potential_outcomes
		elif isinstance(potential_outcomes, (list, tuple)):
			if len(potential_outcomes) == 0:
				raise ValueError('Length of potential_outcomes input is 0')

			potential_outcomes_list = []
			for formula in potential_outcomes:
				if isinstance(formula, str):
					potential_outcomes_list.append(formula)
				else:
					raise TypeError("potential_outcomes argument was of type {}, was expecting type str".format(type(formula).__name__))

			self.potential_outcomes = potential_outcomes_list
		else:
			raise TypeError("potential_outcomes argument was of type {}, was expecting type str".format(type(formula).__name__))

	@staticmethod
	def generate_var(functional_tuple, n, name, **kwargs):
		"""
		A method that takes in a 3-tuple with a method object as the first element,
		and args and kwargs as the second and third elements.

		Returns a pandas Series of size n with the method's output.

		kwargs:

			index: Renames output index, to match with population dataframe
		"""

		func, args, kwargs = functional_tuple

		results = []
		for i in range(n):
			results.append(func(*args, **kwargs))

		output_series = pd.Series(results, name=name)
		if 'index' in kwargs:
			output_series.index = kwargs['index']

		return output_series


	def draw_data(self, n=None, frac=None):
		"""
		Returns a mock dataframe of experiment data generated from declared population and potential_outcomes.

		args:

			N:      Number of rows in returned dataframe. If None and a dataframe has been declared in self.population
					then returns same number of rows as that dataframe. If N is not None and a dataframe has been
					declared, then randomly sample N rows from that generated dataframe. If no dataframe has been
					declared then by default generates 100 rows of observations from dict of variables.
					Cannot be used with frac.
			frac:   Randomly samples this percentage of rows before returning the final dataframe. If None then returns
					generated dataframe as is. Cannot be used with N.

			* Note: frac is applied after N. That is, if N=1000 and frac=0.5 then the returned dataframe will have 500 rows *
		"""

		if self.population is None:
			raise ValueError('A population for this model has not been declared. Use the method declare_population to declared a population.')
		elif isinstance(self.population, dict):
			generated_variable_columns = []
			for variable_name, functional_tuple in self.population.items():
				generated_variable_columns.append(self.generate_var(functional_tuple, 100, variable_name))

			output_df = pd.concat(generated_variable_columns, axis=1)
		elif isinstance(self.population, pd.DataFrame):
			output_df = self.population.copy()
		elif isinstance(self.population, tuple):
			population_df, population_vars = self.population

			output_df = population_df.copy()

			generated_variable_columns = []
			for variable_name, functional_tuple in population_vars.items():
				gen = self.generate_var(functional_tuple, output_df.shape[0], variable_name, index=output_df.index)
				generated_variable_columns.append(gen)

			output_df = pd.concat([output_df] + generated_variable_columns, axis=1)

		if not self.potential_outcomes is None:
			for formula in iter(self.potential_outcomes):
				output_df.eval(formula, inplace=True)

		if (frac is None) & (n is None):
			return output_df
		else:
			return output_df.sample(n=n, frac=frac)


	def __init__(self, **kwargs):
		# If object instantiation has population and/or potential_outcomes arguments
		# use them to update class variables
		if 'population' in kwargs:
			population = kwargs['population']
			self.declare_population(population)
		else:
			self.population = None

		if 'potential_outcomes' in kwargs:
			potential_outcomes = kwargs['potential_outcomes']
			self.declare_potential_outcomes(potential_outcomes)
		else:
			self.potential_outcomes = None