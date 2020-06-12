<!DOCTYPE html>
<html>

<head>
  
  <title>Quick Start - Leaflet</title>

  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  
  <!-- <link rel="shortcut icon" type="image/x-icon" href="docs/images/favicon.ico" /> -->

    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.6.0/dist/leaflet.css" integrity="sha512-xwE/Az9zrjBIphAcBb3F6JVqxf46+CDLwfLMHloNu6KEQCAWi6HcDUbeOfBIptF7tcCzusKFjFw2yuvEpDL9wQ==" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.6.0/dist/leaflet.js" integrity="sha512-gZwIG9x3wUXg2hdXF6+rVkLF/0Vi9U8D2Ntg4Ga5I5BZpVkVxlJWbSQtXPSiUTtC0TjtGOmxa1AJPuV0CPthew==" crossorigin=""></script>

    

    <!-- // These are all PHP variables. The web browser doesn't know about them.

    // $showpoint = $bdd->prepare("select lat,lng from emsdata where (truckid = '83E94E04A08895767DFE0D80A21A07D3') AND (tdate = '20191201');");
    // $showpoint->execute();
    // $nbRows = $showpoint->rowCount();

    // Yes, $donnees is also a PHP variable

    // $donnees = $showpoint->fetch(); -->
  
</head>

<body>

<?php

$servername = "localhost";
$username = "root";
$password = "123456";
$dbname = "fuel_efficiency";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// $showpoint = $conn->prepare("SELECT triggertime, lat, lng FROM emsdata where (truckid = '83E94E04A08895767DFE0D80A21A07D3') AND (tdate = '20191201')");
// $showpoint->execute();
// $nbRows = $showpoint->num_rows;

// $donnees = $showpoint->fetch();

$sql = "SELECT triggertime, lat, lng FROM emsdata where (truckid = '257C0D741E2CDDAFDA1A297FC5AC9964') AND (tdate = '20191201')";
$result = $conn->query($sql);

// if ($result->num_rows > 0) {
//     // output data of each row
//     while($row = $result->fetch_assoc()) {
//         // echo "<br> Time: ". $row["triggertime"]. " - Position: ". $row["lat"]. " " . $row["lng"] . "<br>";
//         $suggestionsArray[] = $row;
//     }
//     $result->close();
// } else {
//     echo "0 results";
// }

//put all of the resulting names into a PHP array
$result_array = Array();
while($name = $result->fetch_assoc()) {
    $result_array[] = $name;
}
//convert the PHP array into JSON format, so it works with javascript
$json_array = json_encode($result_array);

// $conn->close();
?>

<div id="mapid" style="width: 2000px; height: 1000px;"></div>
<script>

  var mymap = L.map('mapid').setView([27, 110], 7);

  L.tileLayer('https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token=pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXVycTA2emYycXBndHRqcmZ3N3gifQ.rJcFIG214AriISLbB6B5aw', {
    maxZoom: 18,
    attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, ' +
      '<a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, ' +
      'Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
    id: 'mapbox/streets-v11',
    tileSize: 512,
    zoomOffset: -1
  }).addTo(mymap);

  var data = <?php echo $json_array; ?>;
	// Note "var i", not "$i". This is the browser iterating through "data".
	for (var i = 0; i < data.length; i++)
	{
		print(data[i].lat,data[i].lng)
	    // Note how "L.marker()" runs only in the browser,
	    // well outside of the <?php ?> tags. PHP doesn't know, nor 
	    // it cares, about Leaflet.
	    // L.marker(data[i].lat, data[i].lng).addTo(mymap);
	    L.circle([data[i].lat, data[i].lng], 200, {
		    color: 'red',
		    fillColor: '#f03',
		    fillOpacity: 1
		  }).addTo(mymap).bindPopup("I am a circle.");

	    // Accessing the properties of the data depends on the structure
	    // of the data. You might want to do stuff like
	    // console.log(data);
	    // while remembering to use the developer tools (F12) in your browser.
	}

  // L.marker([51.5, -0.09]).addTo(mymap)
  //   .bindPopup("<b>Hello world!</b><br />I am a popup.").openPopup();
  // L.marker([28, 109]).addTo(mymap);

  // L.circle([51.508, -0.11], 500, {
  //   color: 'red',
  //   fillColor: '#f03',
  //   fillOpacity: 0.5
  // }).addTo(mymap).bindPopup("I am a circle.");

  // L.polygon([
  //   [51.509, -0.08],
  //   [51.503, -0.06],
  //   [51.51, -0.047]
  // ]).addTo(mymap).bindPopup("I am a polygon.");

  // L.circle([27.508, 110.11], 200, {
  //   color: 'red',
  //   fillColor: '#f03',
  //   fillOpacity: 1
  // }).addTo(mymap).bindPopup("I am a circle.");


  var popup = L.popup();

  function onMapClick(e) {
    popup
      .setLatLng(e.latlng)
      .setContent("You clicked the map at " + e.latlng.toString())
      .openOn(mymap);
  }

  mymap.on('click', onMapClick);

</script>

<?  $conn->close();  ?>

</body>
</html>