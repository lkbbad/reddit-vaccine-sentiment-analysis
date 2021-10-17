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

#  Add a term to the custom dictionary; if it already exists one of
#  the default dictionaries, the request will be ignored unless the
#  user explicitly asks for the value to be changed
#
#  term:     Term to add
#  v:        Valence
#  a:        Arousal
#  replace:  Replace term that exists in default dictionaries

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

#  Return the average arousal for a term
#
#  term:  Term to check (can be string or list of strings)

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

#  Return the raw arousal for a single term
#
#  term:  Term to check

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


def exist( term ):

#  Return True if a term exists in one of the sentiment dictionaries,
#  False otherwise
#
#  term:  Term to check (can be string or list of strings)

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

#  Return the valence and arousal sentiment for a term
#
#  term:  Term to check (can be string or list of strings)

	sen = { 'valence': 0.0, 'arousal': 0.0 }

	if isinstance( term, str ) or isinstance( term, list ):
		sen[ 'valence' ] = valence( term )
		sen[ 'arousal' ] = arousal( term )

	return sen

# End function sentiment


def valence( term ):

#  Return the average valence for a term
#
#  term:  Term to check (can be string or list of strings)

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

#  Return the raw valence for a single term
#
#  term:  Term to check

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
