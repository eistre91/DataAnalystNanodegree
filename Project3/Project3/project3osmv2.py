import xml.etree.cElementTree as ET
import pprint
import re
import codecs
import json
from collections import defaultdict

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

translation_dict = {'av':'Avenue', 'ave': 'Avenue', 'blvd':'Boulevard', 'ctr': 'Center', 'cir': 'Circle',
                    'ct':'Court', 'cres':'Crescent', 'dr':'Drive',
                    'e':'East', 'ln':'Lane', 'pkwy':'Parkway',
                    'pl':'Place', 'rd':'Road', 'st':'Street',
                    'ter':'Terrace', 'terr':'Terrace', 'trl':'Trail',
                    'suit':'Suite', 'ste':'Suite', 'wy':'Way',
                    'w':'West', 'n':'North', 's':'South'}
    
def broad_audit_data(file_in):
  """Prints a broad summary of the structure of the data."""
  tag_occurrence = defaultdict(int)
  tag_attributes_occurrence = defaultdict(int)
  tag_ks_occurrence = defaultdict(int)
  context = ET.iterparse(file_in)
  _, root = next(context)
  for _, element in context:
    tag_occurrence[element.tag] += 1
    for k, v in element.attrib.items():
      tag_attributes_occurrence[(element.tag, k)] += 1
      if element.tag == 'tag' and k == 'k':
        tag_ks_occurrence[element.get('k')] += 1
        if ':' in element.get('k'):
          tag_ks_colon.add(tuple(element.get('k').split(':')))
    root.clear()
    element.clear()
  tag_occurrence_list = list(tag_occurrence.items())
  tag_attributes_occurrence_list = list(tag_attributes_occurrence.items())
  tag_ks_occurrence_list = list(tag_ks_occurrence.items())
  print(type(tag_attributes_occurrence_list))
  print(sorted(tag_occurrence_list, key=lambda x: x[1], reverse=True))
  print(sorted(tag_attributes_occurrence_list, key=lambda x: x[1], reverse=True))
  print(sorted(tag_ks_occurrence_list, key=lambda x: x[1], reverse=True))
  print(sorted([item for item in tag_ks_occurrence_list if (item[1] >= 100)], key=lambda x: x[1], reverse=True))
  
#Things to look at from the first broad audit:
# addr:state
#   Are there different ways of writing Colorado?
#   Are all places properly listed as in Colorado?
#   Audit: Some errorneous values like Denver or zip codes. 
#   Cleaning: Change to state code CO. (OSM prefers that. http://wiki.openstreetmap.org/wiki/Map_Features)
# addr:street
#   Check street names to see how may variations there are
#   Audit: Different ways of writing types of streets.
#          Can safely assume that it's the last word of the string.
#   Cleaning: Pull out the last word of all the strings and rerun a collection.
# addr:postcode
#   Get list of zip codes in the denver area, see if they match
#   See if they're written in different ways
#   Audit: Most are 5 digits.
#          There is a 4 digit which just looks like an error.
#          There's also some in 9 digit format, XXXXX-XXXX.
#          One has Golden, CO as part of the value. Should only contain integers (or '-').
#   Cleaning: Find a reliable list of zips code in Colorado. Check listed ones against that.
#             Trim 9 digit format to be consistent.
#             Remove outliers like 4 digit zip codes.
#             Trim ones with alpha characters down to the integers and check the remaining integers.
#             Valid zip codes are between 80001 and 81658.
# amenity
#   Are there are any amenities with varying names to be consolidated?
#   Audit: Nothing too obvious to consolidate.
#   Cleaning: None.
# natural and man_made
#   Are there multiple ways of denoting true or false?
#   Audit: More descriptive than that. No obvious problems, there are specified values on the wiki.
#   Cleaning: Could check against the wiki list.

def specific_audit_data(file_in):
  """Prints a summary of specific tag values that may require cleaning."""
  addr_state_values = set()
  addr_street_values = set()
  addr_postcode_values = set()
  amenity_values = set()
  natural_values = set()
  man_made_values = set()
  
  context = ET.iterparse(file_in)
  _, root = next(context)
  for _, element in context:
    if element.tag == 'tag':
      k = element.get('k')
      if k == 'addr:state':
        addr_state_values.add(element.get('v'))
      elif k == 'addr:street':
        value = element.get('v').strip()
        if ' ' in value:
          spl = value.split(' ')[-1]
          if any(char.isdigit() for char in spl):
            addr_street_values.add(value)
          elif spl == 'D':
            addr_street_values.add(value)
          else:
            addr_street_values.add(spl)
        else:
          addr_street_values.add(value)
      elif k == 'addr:postcode':
        addr_postcode_values.add(element.get('v'))
      elif k == 'amenity':
        amenity_values.add(element.get('v'))
      elif k == 'natural':
        natural_values.add(element.get('v'))
      elif k == 'man_made':
        man_made_values.add(element.get('v'))
    root.clear()
    element.clear()
  print(addr_state_values)
  print(sorted(addr_street_values))
  print(addr_postcode_values)
  #print(amenity_values)
  #print(natural_values)
  #print(man_made_values)
  
def process_map(file_in, pretty = False):
  """Converts the OSM XML format to a JSON file."""
  file_out = "{0}.json".format(file_in)
  with codecs.open(file_out, "w") as fo:
    context = ET.iterparse(file_in)
    _, root = next(context)  
    for _, element in context:
      el = shape_element(element)
      if el:
        if pretty:
          fo.write(json.dumps(el, indent=2)+"\n")
        else:
          fo.write(json.dumps(el) + "\n")
        root.clear()
        element.clear()
  return 

def shape_element(element):
  """ Takes in an element, cleans it for entry into the database
  and returns the element as a dictionary.
  
    Top level tags for OSM are node, way, relation
      Nodes may contain tag
      Ways may contain nd with ref or tag
      Relation may contain member with type, ref, role or tag 
  """
  node = {}
  # Determine if it's a top level tag. If not, skip this element and return None.
  if element.tag == "node" or element.tag == "way" or element.tag == "relation":
    # Record the xml tag type and set up initial values depending on type.
    node['doc_type'] = element.tag
    node['created'] = {}
    if element.tag == "node":
      node['pos'] = [0, 0]
    elif element.tag == "way":
      node['node_refs'] = []
    elif element.tag == "relation":
      node['member_refs'] = []
    # Record all information contained as attributes in the tag.  
    for k, v in element.attrib.items():
      if k in CREATED:
        node['created'][k] = v
      elif k == "lat":
        node['pos'][0] = float(v)
      elif k == "lon":
        node['pos'][1] = float(v)
      else:
        node[k] = v
    # Record all information in the child tags of the top tag.
    for child in element.iter():
      if child.tag == 'tag':
        k = child.get('k')
        v = child.get('v')
        if problemchars.search(k):
          pass
        # Colons denote more intricate tags. Separate handling depending on what it is.  
        elif ":" in k:
          k_spl = k.split(":")
          # If of the form XXXX:XXXX:XXXX, skip it. I haven't seen any of this form in the data yet.
          # Not likely to be many of them.
          if len(k_spl) > 2:
            pass
          # If of the form addr:XXX.  
          elif k_spl[0] == "addr":
            if 'address' not in node:
              node['address'] = {}
            if k_spl[1] == "state":
              node['address']['state'] = 'CO'
            elif k_spl[1] == "street":
              node['address']['street'] = update_street_name(v)
            elif k_spl[1] == "postcode":  
              cleaned_postal = update_postal_code(v)
              if cleaned_postal != None:
                node['address']['postcode'] = int(v)
            else:
              node['address'][k_spl[1]] = v
          else:
            node["_".join(k_spl)] = v
        else:
          # There's a problem case in which "address" is a separate attribute.
          # Skip if this is the case, else add the information to the node dict.
          if k != 'address':
            node[k] = v
      elif child.tag == 'nd':
        node['node_refs'].append(child.get('ref'))
      elif child.tag == 'member':
        node['member_refs'].append(build_member_dict(child))
    # At this point we've recorded all child information and cleaned if necessary. 
    return node
  else:
    return None
    
def update_street_name(v):
  """Returns the cleaned street name."""
  v_spl = v.split(' ')
  for i, word in enumerate(v_spl):
    # Look for street type appreviations and translate them
    if word.strip('.').lower() in translation_dict:
      v_spl[i] = translation_dict[word.strip('.').lower()]
    # Apply capitalization so that the street name is in camel case
    v_spl[i] = v_spl[i].capitalize()
  return ' '.join(v_spl)
  
def update_postal_code(v):
  """Returns the verified and cleaned postal code or None if invalid."""
  # Pull out the first set of digits if the zip code has a hyphen. 
  if '-' in v:
    v = v.split('-')[0]
  # Remove any non-digit characters.
  v = re.sub("\D", "", v)
  # If the remaining number is a valid zip code, return it.
  if v.isdigit() and 80001 <= int(v) and int(v) <= 81658:
    return int(v)
  else:
    return None
    
def build_member_dict(child):
  """Build the member dictionary that goes with the relation element."""
  member_dict = {}
  member_dict['ref'] = child.get('ref')
  member_dict['role'] = child.get('role')
  member_dict['type'] = child.get('type')
  return member_dict
  
# Cleaning Plan 1:
#   Make all state values CO. - done
#   Test all zip code values to be between 80001 and 81658. - done
#     Remove hyphenated zip code and leave first five numbers. - done
#     Remove all alpha characters. - done
#       Check remaining zip code by above criteria. - done
#   Apply common translations to space seperated abbreviations for the street names. - done
#   Write street names in camel case. - done
  
# Street names
# Postal Codes
    # Inconsistent - Different writing conventions
    # Incorrect - Wouldn't match with outside more reliable database
  
# Is the data valid? 
  # The XML appears to be within the structure it's expected to be. It doesn't deviate as far as I can tell. 
# Is the data accurate? 
  # What are some outside comparisons I can do to determine this? Google maps?
  # Googling in general.
  # Fixing up TIGER mistakes: http://wiki.openstreetmap.org/wiki/TIGER_fixup
# Is the data complete?
  # Again, comparing to another database.
  # Determine if there's some data that really should be there that's missing. Like a user recorded for every node.
# Is the data consistent?
  # Zip codes like above, see if they're written different.
  # Street names written differently. 
  # Look for other inconsistencies in frequent fields.
# Is the data uniform?
  # Are there distance measurements somewhere that seem out of whack? 
  
if __name__ == "__main__":
    #broad_audit_data('sample.osm')
    
    #specific_audit_data('sample.osm')

    #broad_audit_data('denver-boulder_colorado.osm')
    
    #specific_audit_data('denver-boulder_colorado.osm')
    
    process_map('sample.osm')