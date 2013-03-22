// Init the map
var map = new OpenLayers.Map(
  'map',
  {
    projection: "EPSG:3857"
  }
);
var bathymetry = new OpenLayers.Layer.WMS(
  "Bathymetry", "", {layers: "basic", transparent: "true"}
)
map.addLayer(bathymetry);
var depth = new OpenLayers.Layer.WMS(
  "Depth", "", {layers: "basic", transparent: "true"}
)
map.addLayer(depth);
var grid = new OpenLayers.Layer.WMS(
  "Grid", "", {layers: "basic", transparent: "true"}
)
map.addLayer(grid);
var osm = new OpenLayers.Layer.OSM()
map.addLayer(osm);

// Functions
function updateLayer(){
  updateGrid();
  updateBathymetry();

  var layer = $('select#layer option:selected').val()
  grid.setUrl("/3di/wms?layer=" + layer + ":grid" + "&time=" + time);
  grid.redraw()
  bathymetry.setUrl("/3di/wms?layer=" + layer + ":bathymetry" + "&time=" + time);
  bathymetry.redraw()
  // Determine bounds
  $.ajax(
    '/3di/wms',
    { 
      data: {
        request: 'getinfo',
        layer: layer,
        srs: 'epsg:3857'
      },
      success: updateLayerFromData
    }
  );
}

function updateLayerFromData(data) {
  $("#slider").slider("option", "max", data['timesteps'] - 1);
  updateDepth();
  var bounds = data['bounds'];
  map.zoomToExtent(
    new OpenLayers.Bounds(bounds[0], bounds[1], bounds[2], bounds[3])
  )
}

function updateDepth(){
  // Get variables from dom
  var layer = $('select#layer option:selected').val();
  var time = $('#slider').slider("option", "value")
  if ($("input#antialias").is(":checked")) {
    var antialias = '2'
  } else {
    var antialias = '1'
  }
  // Update label
  $("#time").text(time);
  // Set and redraw layer
  var url = "/3di/wms?layer=" + layer + ":depth";
  url += "&time=" + time + "&antialias=" + antialias;
  depth.setUrl(url);
  depth.redraw();
}

function updateBathymetry(){
  var layer = $('select#layer option:selected').val();
  if ($("input#antialias").is(":checked")) {
    var antialias = '2'
  } else {
    var antialias = '1'
  }
  var url = "/3di/wms?layer=" + layer + ":bathymetry";
  url += "&antialias=" + antialias;
  bathymetry.setUrl(url);
  bathymetry.redraw()
}

function updateGrid(){
  var layer = $('select#layer option:selected').val();
  if ($("input#antialias").is(":checked")) {
    var antialias = '2'
  } else {
    var antialias = '1'
  }
  var url = "/3di/wms?layer=" + layer + ":grid";
  url += "&antialias=" + antialias;
  grid.setUrl(url);
  grid.redraw();
}

function toggleGrid(){
  var state = $("input#grid").is(":checked");
  grid.setVisibility(state);
}
function toggleDepth(){
  var state = $("input#depth").is(":checked");
  depth.setVisibility(state);
}
function toggleBathymetry(){
  var state = $("input#bathymetry").is(":checked");
  bathymetry.setVisibility(state);
}
function toggleOsm(){
  var state = $("input#osm").is(":checked");
  osm.setVisibility(state);
}
function toggleAntialias(){
  updateBathymetry();
  updateDepth();
  updateGrid();
}
  

// Slider
function slide(ui, slider){
  updateDepth();
}

$("#slider").slider({
  min: 0,
  max: 143,
  slide: slide
});

// Bind controls
$("select#layer").on("change", updateLayer);
$("select#mode").on("change", updateLayer);
$("input#grid").on("change", toggleGrid);
$("input#depth").on("change", toggleDepth);
$("input#bathymetry").on("change", toggleBathymetry);
$("input#osm").on("change", toggleOsm);
$("input#antialias").on("change", toggleAntialias);

toggleGrid()
toggleBathymetry()
updateLayer();
