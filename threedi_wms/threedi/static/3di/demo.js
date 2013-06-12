/* global OpenLayers, $, console */
/* jshint strict: false */

// Init the map
var map = new OpenLayers.Map(
  'map',
  {
    projection: "EPSG:3857"
  }
);
var bathymetry = new OpenLayers.Layer.WMS(
  "Bathymetry", "", {layers: "basic", transparent: "true"}
);
map.addLayer(bathymetry);
var depth = new OpenLayers.Layer.WMS(
  "Depth", "", {layers: "basic", transparent: "true"}
);
map.addLayer(depth);
var flood = new OpenLayers.Layer.WMS(
  "Flood", "", {layers: "basic", transparent: "true"}
);
map.addLayer(flood);
var grid = new OpenLayers.Layer.WMS(
  "Grid", "", {layers: "basic", transparent: "true"}
);
map.addLayer(grid);
var osm = new OpenLayers.Layer.OSM();
map.addLayer(osm);

var info;

// Dom access functions
function getAntialias(){
  if ($("input#antialias").is(":checked")) {
    return 'yes';
  } else {
    return 'no';
  }
}
function getWaves(){
  if ($("input#waves").is(":checked")) {
    return '&anim_frame=0';
  } else {
    return '';
  }
}
function getLayer(){
  return $('select#layer option:selected').val();
}
  
// Updaters
function updateLayer(){
  // Determine bounds
  $.ajax(
    '/3di/wms',
    { 
      data: {
        request: 'getinfo',
        layers: getLayer(),
        srs: 'epsg:3857'
      },
      success: updateInfo
    }
  );
}
function updateInfo(data){
  // There'
  info = data;
  updateSlider();
  updateGrid();
  updateDepth();
  updateFlood();
  updateBathymetry();
  var bounds = data.bounds;
  map.zoomToExtent(
    new OpenLayers.Bounds(bounds[0], bounds[1], bounds[2], bounds[3])
  );
}

function updateGrid(){
  var url = "/3di/wms";
  url += "?LAYERS=" + getLayer() + ":grid";
  url += "&antialias=" + getAntialias();
  grid.setUrl(url);
  grid.redraw();
}

function updateDepth(){
  var url = "/3di/wms";
  url += "?LAYERS=" + getLayer() + ":depth";
  url += "&time=" + getTime();
  url += "&antialias=" + getAntialias();
  url += "&nocache=yes";
  url += getWaves();
  console.log(url);
  depth.setUrl(url);
  depth.redraw();
}

function updateFlood(){
  var url = "/3di/wms";
  url += "?LAYERS=" + getLayer() + ":flood";
  url += "&time=" + getTime();
  url += "&antialias=" + getAntialias();
  url += "&nocache=yes";
  url += getWaves();
  console.log(url);
  flood.setUrl(url);
  flood.redraw();
}

function updateBathymetry(){
  bathymetry.redraw();
  var url = "/3di/wms";
  url += "?LAYERS=" + getLayer() + ":bathymetry";
  url += "&antialias=" + getAntialias();
  url += "&limits=" + info.limits[0] + "," + info.limits[1];
  bathymetry.setUrl(url);
  bathymetry.redraw();
}

function toggleGrid(){
  var state = $("input#grid").is(":checked");
  grid.setVisibility(state);
}
function toggleDepth(){
  var state = $("input#depth").is(":checked");
  depth.setVisibility(state);
}
function toggleFlood(){
  var state = $("input#flood").is(":checked");
  flood.setVisibility(state);
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
  updateGrid();
  updateDepth();
  updateFlood();
  updateBathymetry();
}
function toggleWaves(){
  updateDepth();
  updateFlood();
}
  
// Slider
function slide(ui, slider){
  $("#time").text(slider.value);
  updateDepth();
  updateFlood();
}
function getTime(){
  return $("#time").text();
}
function setTime(time){
  $('#slider').slider("option", "value", time);
  $("#time").text(time);
}
function updateSlider() {
  var sliderMax = info.timesteps - 1;
  $("#slider").slider("option", "max", sliderMax);
  if (getTime() > sliderMax) {setTime(sliderMax);}
}
$("#slider").slider({
  min: 0,
  max: 143,
  slide: slide
});

// Bind controls
$("select#layer").on("change", updateLayer);
$("input#grid").on("change", toggleGrid);
$("input#depth").on("change", toggleDepth);
$("input#flood").on("change", toggleFlood);
$("input#bathymetry").on("change", toggleBathymetry);
$("input#osm").on("change", toggleOsm);
$("input#antialias").on("change", toggleAntialias);
$("input#waves").on("change", toggleWaves);

toggleGrid();
toggleFlood();
toggleBathymetry();
updateLayer();

// Click handler
OpenLayers.Control.Click = OpenLayers.Class(OpenLayers.Control, {
  defaultHandlerOptions: {
    'single': true,
    'double': false,
    'pixelTolerance': 0,
    'stopSingle': false,
    'stopDouble': false
  },
  initialize: function() {
    this.handlerOptions = OpenLayers.Util.extend(
      {}, this.defaultHandlerOptions
    );
    OpenLayers.Control.prototype.initialize.apply(
      this, arguments
    ); 
    this.handler = new OpenLayers.Handler.Click(
      this, {
        'click': this.trigger
      }, this.handlerOptions
    );
  }, 
  trigger: function(e) {
    var lonlat = map.getLonLatFromPixel(e.xy);
    console.log([lonlat.lat, lonlat.lon]);
    $.ajax(
      '/3di/data',
      { 
        data: {
          request: 'gettimeseries',
          layers: getLayer(),
          srs: map.getProjection(),
          point: lonlat.lon.toString() + ',' + lonlat.lat.toString(),
        },
        success: function(data) {console.log(data);}
      }
    );
    }
});

var click = new OpenLayers.Control.Click();
map.addControl(click);
click.activate();
