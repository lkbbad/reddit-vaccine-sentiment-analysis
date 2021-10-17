#!/usr/bin/python

#- SENTIMENT.PY ------------------------------------------------------------#
#  Routines to calulate average valence and arousal for one or more terms	#
#  using the ANEW and Happiness sentiment dictionaries						#
#																			#
#- Modification History: ---------------------------------------------------#
#  When:		Who:					Comments:							#
#																			#
#  28-Sep-14	Christopher G. Healey	Converted from Javascript			#
#  17-Dec-17	Christopher G. Healey	Changed to SENTIMENT.PY to clarify  #
#										these are not ANEW-only terms		#
#  09-Oct-20	Christopher G. Healey	Added describe() to provide brief	#
#										text description for given valence  #
#										and arousal							#
#---------------------------------------------------------------------------#

import math
import nltk

#  Import raw ANEW and Happiness dictionary data from term file

from .sentiment_term import anew_word as anew_word
from .sentiment_term import anew_stem as anew_stem
from .sentiment_term import hapi_word as hapi_word

#  Setup a "custom" dictionary to allow users to extend the ANEW and
#  happiness dictionaries

cust_dict = { }							# Custom dictionary, raw terms
cust_stem = { }							# Custom dictionary, stemmed terms


def add_term( term, v, a, replace = False ):
	"""Add a term to the custom dictionary; if it already exists one of
	the default dictionaries, the request will be ignored unless the
	user explicitly asks for the value to be changed

	Args:
		term (string): Term to add
		v (float): Term valence
		a (float): Term arousal
		replace (bool, optional): Replace existing term. Defaults to False.

	Returns:
		None: Return directly if term exists and replace is False
	"""

	global cust_dict
	global cust_stem


	#  If term already exists and user does not ask to replace it, stop

	if exist( term ) and replace != True:
		return

	#  Otherwise either replace it or add it to the custom dictionary

	if term in anew_word and replace == True:
		anew_word[ term ][ 'avg' ][ 0 ] = a
		anew_word[ term ][ 'avg' ][ 1 ] = v
	elif term in anew_stem and replace == True:
		anew_stem[ term ][ 'avg' ][ 0 ] = a
		anew_stem[ term ][ 'std' ][ 1 ] = v
	elif term in hapi_word:
		hapi_word[ term ][ 'avg' ][ 0 ] = a
		hapi_word[ term ][ 'std' ][ 1 ] = v
	else:
		cust_dict[ term ] = { }
		cust_dict[ term ][ 'dict' ] = "custom"
		cust_dict[ term ][ 'word' ] = term
		cust_dict[ term ][ 'avg' ] = [ a, v ]
		cust_dict[ term ][ 'std' ] = [ 1, 1 ]
		cust_dict[ term ][ 'fq' ] = 1

		#  Build a stem for the custom term

		porter = nltk.stem.porter.PorterStemmer()
		stem = porter.stem( term )

		cust_dict[ term ][ 'stem' ] = porter.stem( term )

		#  Add term to custom stem dictionary with stem as key

		cust_stem[ stem ] = { }
		cust_stem[ stem ][ 'dict' ] = "custom"
		cust_stem[ stem ][ 'word' ] = stem
		cust_stem[ stem ][ 'stem' ] = stem
		cust_stem[ stem ][ 'avg' ] = [ a, v ]
		cust_stem[ stem ][ 'std' ] = [ 1, 1 ]
		cust_stem[ stem ][ 'fq' ] = 1

#  End function add_term


def arousal( term ):
	"""Return the average arousal for a term

	Args:
		term (string or [string]): Term(s) to query

	Returns:
		float or [float]: Arousal of term, if it/they exists
	"""

	if isinstance( term, str ):
		return arousal_raw( term )[ 0 ]

	elif not isinstance( term, list ):
		return 0.0

	#  At this point we know we're working with a list of terms

	c = 2.0 * math.pi
	prob = [ ]
	prob_sum = 0.0
	a_mu = [ ]

	for t in term:
		if exist( t ):
			a = arousal_raw( t )

			p = 1.0 / math.sqrt( c * math.pow( a[ 1 ], 2.0 ) )
			prob.append( p )
			prob_sum = prob_sum + p
		
			a_mu.append( a[ 0 ] )

	arousal = 0.0
	for i in range( 0, len( a_mu ) ):
		arousal = arousal + ( prob[ i ] / prob_sum * a_mu[ i ] )

	return arousal

# End function arousal


def arousal_raw( term ):
	"""Return the raw arousal for a single term

	Args:
		term (string): Term to query

	Returns:
		float: Term's arousal, if term exists
	"""

	global cust_dict
	global cust_stem


	if not exist( term ):
		avg = 0.0
		std = 0.0
	elif term in anew_word:
		avg = anew_word[ term ][ 'avg' ][ 1 ]
		std = anew_word[ term ][ 'std' ][ 1 ]
	elif term in anew_stem:
		avg = anew_stem[ term ][ 'avg' ][ 1 ]
		std = anew_stem[ term ][ 'std' ][ 1 ]
	elif term in cust_dict:
		avg = cust_dict[ term ][ 'avg' ][ 1 ]
		std = cust_dict[ term ][ 'std' ][ 1 ]
	elif term in cust_stem:
		avg = cust_stem[ term ][ 'avg' ][ 1 ]
		std = cust_stem[ term ][ 'std' ][ 1 ]
	else:
		avg = hapi_word[ term ][ 'avg' ][ 1 ]
		std = hapi_word[ term ][ 'std' ][ 1 ]

	return [ avg, std ]

# End function arousal_raw


def describe( *args, **kwds ):
	if len( args ) == 1 and isinstance( args[ 0 ], str ):
		return describe_term( args[ 0 ] )
	elif len( args ) == 1 and isinstance( args[ 0 ], list ):
		return describe_term( args[ 0 ] )
	elif len( args ) == 2:
		rc = [ False, False ]
		rc[ 0 ] = isinstance( args[ 0 ], int ) or isinstance( args[ 0 ], float )
		rc[ 1 ] = isinstance( args[ 1 ], int ) or isinstance( args[ 1 ], float )
		if rc[ 0 ] and rc[ 1 ]:
			return describe_sentiment( args[ 0 ], args[ 1 ] )
		else:
			print( 'describe(), argument must be string, list of strings, or float,float' )
			return 'unknown'
	else:
		print( 'describe(), argument must be string, list of strings, or float,float' )
		return 'unknown'

#  End method describe


def describe_term( term ):
	"""Take a term, then convert it to valence and arousal, and then to a
	Russell-like description with a modifier of somewhat, moderately, just
	the term alone, or very

	Args:
		term (string or [string]): Term(s) to query

	Returns:
		string:  modifier+term
	"""

	sen = sentiment( term )

	if sen[ 'arousal' ] == 0:					#  unknown term(s)?
		return 'unknown'
	else:
		return describe_sentiment( sen[ 'valence' ], sen[ 'arousal' ] )

# End function describe_term


def describe_sentiment( v, a ):
	"""Take a valence and arousal on the ANEW range 1..9, then try to convert
	it to a Russell-like description with a modifier of somewhat, moderately,
	just the term alone, or very
	
	Arguments:
		v (float):  Valence on range 1..9
		a (float):  Arousal on range 1..9
		
	Returns:
		string:  modifier+term
	"""
	
	#  Check for valid arguments	

	if v < 1.0 or v > 9.0 or a < 1.0 or a > 9.0:
		print( 'describe(), valence/arousal must be on range 1..9' )
		return( 'unknown' )
	
	#  Center of circumplex (5,5) will give an r=0, div by zero error, so handle
	#  explicitly
	
	if v == 5.0 and a == 5.0:
		return 'average'
	
	#  Angular cutoffs for different emotional states (same on top and bottom)

	ang = [ 0, 18.43, 45, 71.57, 90, 108.43, 135, 161.57, 180 ]
	
	#  Terms to return for bottom, top half of circumplex

	lower_term =[
		'contented', 'serene', 'relaxed', 'calm',
		'bored', 'lethargic', 'depressed', 'sad'
	]
	upper_term = [
		'happy', 'elated', 'excited', 'alert',
		'tense', 'nervous', 'stressed', 'upset'
	]
	
	#  Normalize valence and arousal, use polar coordinates to get angle
	#  clockwise along bottom, counterclockwise along top
	
	v = ( ( v - 1.0 ) - 4.0 ) / 4.0
	a = ( ( a - 1.0 ) - 4.0 ) / 4.0	
	r = math.sqrt( ( v ** 2.0 ) + ( a ** 2.0 ) )
	direction = math.degrees( math.acos( v / r ) )
	
	#  Normalize radius for "strength" of emotion
	
	if direction <= 45 or direction >= 135:
		r = r / math.sqrt( ( a ** 2.0 ) + 1.0 )
	else:
		r = r / math.sqrt( ( v ** 2.0 ) + 1.0 )
		
	#  Based on normalized radius, choose four possible modifier words
		
	if r <= 0.25:
		modify = 'slightly '
	elif r <= 0.5:
		modify = 'moderately '
	elif r <= 0.75:
		modify = ''
	else:
		modify = 'very '
	
	#  Use normalized arousal to determine if we're on bottom of top
	#  of circumplex
	
	if a < 0:
		term = lower_term
	else:
		term = upper_term
		
	#  Walk along angular boundaries until we determine which "slice"
	#  our valence and arousal point lies in, return corresponding term
		
	for i in range( 0, len( term ) ):
		if direction >= ang[ i ] and direction <= ang[ i + 1 ]:
			return( modify + term[ i ] )

	#  If somehow we fall through the loop, which should never happen,
	#  return unknown

	print( 'describe(), unexpected angle', a, 'did not match any term' )
	return 'unknown'

#  End function describe_sentiment


def exist( term ):
	"""Return True if a term exists in one of the sentiment dictionaries,
	False otherwise

	Args:
		term (string or [string]): Term(s) to query

	Returns:
		bool or [bool]: Term exists (True) or does not exist (False)
	"""

	global cust_dict
	global cust_stem


	if isinstance( term, str ):
		ex = term in anew_word or term in anew_stem or\
		     term in hapi_word or term in cust_dict or term in cust_stem
		return ex

	elif isinstance( term, list ):
		term_list = [ ]

		for t in term:
			ex = t in anew_word or t in anew_stem or\
			     t in hapi_word or t in cust_dict or t in cust_stem
			term_list.append( ex )

		return term_list

	else:
		return False

# End function exist


def sentiment( term ):
	"""Return the valence and arousal sentiment for a term

	Args:
		term (string or [string]): Term(s) to query

	Returns:
		dict: {'valence','arousal'} for term(s)
	"""

	sen = { 'valence': 0.0, 'arousal': 0.0 }

	if isinstance( term, str ) or isinstance( term, list ):
		sen[ 'valence' ] = valence( term )
		sen[ 'arousal' ] = arousal( term )

	return sen

# End function sentiment


def valence( term ):
	"""Return the average valence for a term

	Args:
		term (string or [string]): Term(s) to query

	Returns:
		float or [float]: Valence of term(s), if it/they exists
	"""

	if isinstance( term, str ):
		return valence_raw( term )[ 0 ]

	elif not isinstance( term, list ):
		return 0.0

	#  At this point we know we're working with a list of terms

	c = 2.0 * math.pi
	prob = [ ]
	prob_sum = 0.0
	v_mu = [ ]

	for t in term:
		if exist( t ):
			v = valence_raw( t )

			p = 1.0 / math.sqrt( c * math.pow( v[ 1 ], 2.0 ) )
			prob.append( p )
			prob_sum = prob_sum + p
		
			v_mu.append( v[ 0 ] )

	val = 0.0
	for i in range( 0, len( v_mu ) ):
		val = val + ( prob[ i ] / prob_sum * v_mu[ i ] )

	return val

# End function valence


def valence_raw( term ):
	"""Return the raw valence for a single term

	Args:
		term (string): Term to query

	Returns:
		float: Term's valence, if term exists
	"""

	global cust_dict
	global cust_stem


	if not exist( term ):
		avg = 0.0
		std = 0.0
	elif term in anew_word:
		avg = anew_word[ term ][ 'avg' ][ 0 ]
		std = anew_word[ term ][ 'std' ][ 0 ]
	elif term in anew_stem:
		avg = anew_stem[ term ][ 'avg' ][ 0 ]
		std = anew_stem[ term ][ 'std' ][ 0 ]
	elif term in cust_dict:
		avg = cust_dict[ term ][ 'avg' ][ 0 ]
		std = cust_dict[ term ][ 'std' ][ 0 ]
	elif term in cust_stem:
		avg = cust_stem[ term ][ 'avg' ][ 0 ]
		std = cust_stem[ term ][ 'std' ][ 0 ]
	else:
		avg = hapi_word[ term ][ 'avg' ][ 0 ]
		std = hapi_word[ term ][ 'std' ][ 0 ]

	return [ avg, std ]

# End function valence_raw