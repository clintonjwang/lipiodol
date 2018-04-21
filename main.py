import argparse
import easygui
import lipiodol_methods as lm
import lipiodol_vis as lvis

def get_inputs_gui():
	"""UI flow. Returns None if cancelled or terminated with error,
	else returns [user, pw, acc_nums, save_dir]."""
	if not easygui.msgbox(('This utility downloads studies from the YNHH VNA. It can be queried by accession number, '
						'MRN or study keyword. It saves each study to its own folder, with each series stored in a '
						'separate subfolder.\n')):
		return None

	try:
		options = {}

		fieldValues = easygui.multpasswordbox(msg='Enter credentials to access VNA.', fields=["Username", "Password"])
		if fieldValues is None:
			return None
		user, pw = fieldValues

		msg = "How do you want to query studies?"
		choices = ["Accession Numbers", "Patient MRNs", "Study Keywords"]
		reply = easygui.buttonbox(msg, msg, choices=choices)
		if reply == choices[0]:
			options['search_type'] = "accnum"
			options['review'] = False
		elif reply == choices[1]:
			options['search_type'] = "mrn"
			options['review'] = True
		elif reply == choices[2]:
			options['search_type'] = "keyword"
			options['review'] = True
		else:
			return None

		msg = "Enter query parameters (only " + reply + " is mandatory)"
		title = "Query parameters"
		fieldNames = [reply + " separated by commas or spaces",
					"Start date (YYYYMMDD format)",
					"End date (YYYYMMDD format)",
					"Modality to limit the search to (use the 2-letter code used in DICOM, i.e. MR, CR, etc.)",
					"Keywords for series to exclude if they appear in the description (e.g. sub, localizer, cor)"]#,
					#"Keywords for series to include if they appear in the description"]
		fieldValues = easygui.multenterbox(msg, title, fieldNames)

		while True:
			if fieldValues is None:
				return None

			errmsg = ""
			if fieldValues[0].strip() == "":
				errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[0])
			elif len(fieldValues[1].strip()) > 8:
				errmsg = errmsg + ('"%s" is not in YYYYMMDD format.\n\n' % fieldValues[1])
			elif len(fieldValues[2].strip()) > 8:
				errmsg = errmsg + ('"%s" is not in YYYYMMDD format.\n\n' % fieldValues[2])
			elif len(fieldValues[3].strip()) > 2:
				errmsg = errmsg + ('"%s" is not in the DICOM 2-letter format.\n\n' % fieldValues[3])

			if errmsg == "":
				break # no problems found

			fieldValues = easygui.multenterbox(errmsg, title, fieldNames, fieldValues)

		query_terms = fieldValues[0].replace(',', ' ').split()
		options['start_date'] = _parse_field_value(fieldValues[1])
		options['end_date'] = _parse_field_value(fieldValues[2])
		options['modality'] = _parse_field_value(fieldValues[3])
		options['exclude_terms'] = fieldValues[4].replace(',', ' ').split()
		#options['include_terms'] = fieldValues[5].replace(',', ' ').split()
		options['verbose'] = True
		options['keep_phi'] = False

		options['save_dir'] = easygui.diropenbox(msg='Select a folder to save your images to.')
		if options['save_dir'] is None:
			return None

		if len(os.listdir(options['save_dir'])) > 0:
			options['overwrite'] = easygui.ynbox("Overwrite any existing folders?")
			if options['overwrite'] is None:
				return None

	except:
		easygui.exceptionbox()
		return None

	return [user, pw, query_terms, options]

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Analyzes BL MR and 24h CT for a patient receiving cTACE.')
	parser.add_argument('--mrbl', help='DICOM directory for baseline MR')
	parser.add_argument('--ct24', help='DICOM directory for 24h CT')
	args = parser.parse_args()

	s = time.time()
	img, D = hf.dcm_load(args.mrbl, True, True)
	print("Time to convert dcm to npy: %s" % str(time.time() - s))

	s = time.time()
	img, D = hf.dcm_load(args.ct24, True, True)
	lm.seg_target_lipiodol(img)
	print("Time to load voi coordinates: %s" % str(time.time() - s))

	print("Finished!")