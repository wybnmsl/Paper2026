<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN"
        "http://www.w3.org/TR/html4/strict.dtd">
<html lang="en">
<head>
	<meta http-equiv="content-type" content="text/html; charset=utf-8">
	<title>Rice University -- Web Services</title>
	<!-- 
			Rice University Web Services Form Kit
			is copyright 2008 by Rice University but
			makes use of the following packages:
			
			jQuery 1.2.6 (http://jquery.com) and jQuery UI 
			by John Resig is used here under the MIT license found in
			js/jquery_license.txt
			
			jQuery Validation by Jörn Zaefferer
			(http://bassistance.de/jquery-plugins/jquery-plugin-validation/)
			is used here under the MIT license found in
			js/jquery_validate_license.txt
			
			jQuery Tablesorter by Christian Bach
			(http://tablesorter.com/docs/) is used here
			under the MIT license found in
			js/jquery_tablesorter_license.txt
			
			jQuery Datepicker by Marc Grabanski
			(http://marcgrabanski.com/pages/code/jquery-ui-datepicker)
			is used here under the MIT license found in
			js/jquery_datepicker_license.txt
			
			jQuery Timepickr, Dropslide, Haineault-Utils,
			and Haineault-Strings by Maxime Haineault
			are used here under the terms of the MIT license
			found in js/jquery_haineault_license.txt
			
			jQuery Combobox by Peter of the Norse
			(http://rahmcoff.livejournal.com/12071.html) 
			is used here informally in a pre-release bug-testing version
			until it is released under typical jQuery licensing
			
			960 Grid System (http://960.gs/) by Nathan Smith
			is used here under the MIT license found in
			css/960_license.txt
			
			Silk icon set (http://www.famfamfam.com/lab/icons/silk/)
			by Mark James is used here under a Creative Commons 2.5
			Attribution License.  The full list of icons can be
			found in img/icons/index_abc.png
	-->
	<link rel="stylesheet" type="text/css" href="css/960_reset.css">
	<link rel="stylesheet" type="text/css" href="css/960_text.css">
	<link rel="stylesheet" type="text/css" href="css/960_grid.css">
	<link rel="stylesheet" type="text/css" href="css/rice_text.css">
	<link rel="stylesheet" type="text/css" href="css/rice_regions.css">
	<link rel="stylesheet" type="text/css" href="css/rice_forms.css">
	<link rel="stylesheet" type="text/css" href="css/site.css">
	<link rel="stylesheet" type="text/css" href="css/jquery_datepicker.css">
	<link rel="stylesheet" type="text/css" href="css/jquery_timepickr.css">
	<script type="text/javascript" src="js/jquery-1.2.6.min.js"></script>
	<script type="text/javascript" src="js/jquery.ui-1.5.2/ui/ui.core.js"></script>
	<script type="text/javascript" src="js/jquery.validate.min.js"></script>
	<script type="text/javascript" src="js/jquery.validate.extras.js"></script>
	<script type="text/javascript" src="js/jquery.tablesorter.min.js"></script>
	<script type="text/javascript" src="js/jquery.ui.datepicker.js"></script>
	<script type="text/javascript" src="js/jquery.timepickr.js"></script>
	<script type="text/javascript" src="js/jquery.combobox.js"></script>
	<!-- date picker i liked -->
	<!-- resizeable text area -->
	<script type="text/javascript">
		$(document).ready(function () {
			$("input[name=search]").focus(function () {
				if ($("input[name=search]").val() == 'Search all of Rice') {
					$("input[name=search]").val('');
				}
			});
			$("input[name=date]").datepicker({showOn: "both", buttonImage: "img/icons/calendar_view_month.png", buttonImageOnly: true});
			$("input[name=time]").timepickr({handle: "#time_target", suffix: ['AM', 'PM'], convention: 12});
            // temporary fix..
            $('.ui-dropslide ol:eq(0) li:first').mouseover();
            // apply theme
            $("input[name=time]").next().addClass('dark');
		});
	</script>
</head>
<body>
	<div class="container_12">
		<div id="rice_header" class="rice_text">
			<div id="rice_logo">
				<a href="http://www.rice.edu/"><img src="img/rice/logo_rice.gif" alt="Rice Unconventional Wisdom" border="0"></a>
			</div>
			<div id="rice_title">
				<a href="http://webservices.rice.edu">Web Services</a>
			</div>
			<div id="rice_search">
				<form name="rice_google_search" action="http://www.rice.edu/Internet/search/query.php" method="get">
					<input class="searchtext" value="Search all of Rice" name="search" valign="top" id="q" size="15" maxlength="256" type="text">
					<input name="tab" value="Web" type="hidden">
					<input name="xtra_search" value="" type="hidden">
					<input name="submit" class="searchbutton" value="Web" src="img/rice/search.gif" alt="Search" type="image">
				</form>
			</div>
		</div>
		<div class="clear">&nbsp;</div>
		<div class="grid_12">
			<h2>Not found ...</h2>
                        <p>The resource you requested cannot be found.  Please be certain the URL is correct, or contact the owner of the resource if you need more information about where it resides.</p>
		</div>
			<div class="clear">&nbsp;</div>
			<div id="rice_footer" class="rice_text">
				<a href="http://www.rice.edu/it"><img src="img/rice/logo_it.gif" alt="Information Technology" border="0"></a>
				<a href="http://www.rice.edu/it">Information Technology</a> 6100 Main, Houston, Texas 77005-1827 &nbsp; &nbsp; 
				&nbsp; Mailing Address: P.O. Box 1892, Houston, Texas 77251-1892<br><br>
				&copy; 2008 Rice University <a href="http://www.rice.edu/">Rice Home</a><a href="http://events.rice.edu/">University Calendar</a>
				<a href="http://www.rice.edu/search/query.php?tab=sitemap">Rice A-Z</a>
				<a href="http://www.rice.edu/maps">Maps</a>
				<a href="mailto:helpdesk@rice.edu">Feedback</a>
			</div>
		</div>
	</div>
</body>
</html>
