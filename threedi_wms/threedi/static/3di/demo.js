/* global OpenLayers, $, console */
/* jshint strict: false */

// Init the map
var map = new OpenLayers.Map(
  'map',
  {
    projection: "EPSG:3857"
  }
);

// Disable default double click zoom
var Navigation = new OpenLayers.Control.Navigation({
  defaultDblClick: function() { return; }
});
map.addControl(Navigation);

var bathymetry = new OpenLayers.Layer.WMS(
  "Bathymetry", "", {layers: "basic", transparent: "true"}
);
map.addLayer(bathymetry);
var depth = new OpenLayers.Layer.WMS(
  "Depth", "", {layers: "basic", transparent: "true"}
);
map.addLayer(depth);
var velocity = new OpenLayers.Layer.WMS(
  "Velocity", "", {layers: "basic", transparent: "true"}
);
map.addLayer(velocity);
var flood = new OpenLayers.Layer.WMS(
  "Flood", "", {layers: "basic", transparent: "true"}
);
map.addLayer(flood);
var grid = new OpenLayers.Layer.WMS(
  "Grid", "", {layers: "basic", transparent: "true"}
);
map.addLayer(grid);
var quad_grid = new OpenLayers.Layer.WMS(
    "QuadGrid", "", {layers: "basic", transparent: "true"}
);
map.addLayer(quad_grid);
var osm = new OpenLayers.Layer.OSM();
map.addLayer(osm);

var info;
var lastClickPoint;

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

function getMessages(){
    return $('input#messages').is(":checked") + "&random=" + Math.random() ;
}
  

function getInterpolate(){
    return $('select#interpolate option:selected').val();
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
  updateQuadGrid();
  updateDepth();
  updateFlood();
  updateBathymetry();
  updateVelocity();
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
function updateQuadGrid(){
    var url = "/3di/wms";
    url += "?LAYERS=" + getLayer() + ":quad_grid";
    url += "&antialias=" + getAntialias();
    quad_grid.setUrl(url);
    quad_grid.redraw();
}

function updateDepth(){
  var url = "/3di/wms";
  url += "?LAYERS=" + getLayer() + ":depth";
  url += "&time=" + getTime();
  url += "&antialias=" + getAntialias();
  url += "&messages=" + getMessages();
  url += "&interpolate=" + getInterpolate();
  url += "&nocache=yes";
  url += getWaves();
  depth.setUrl(url);
  depth.redraw();
  console.log(url);
}

function updateFlood(){
  var url = "/3di/wms";
  url += "?LAYERS=" + getLayer() + ":flood";
  url += "&time=" + getTime();
  url += "&antialias=" + getAntialias();
  url += "&messages=" + getMessages();
  url += "&nocache=yes";
  url += getWaves();
  flood.setUrl(url);
  flood.redraw();
}

function updateBathymetry(){
  bathymetry.redraw();
  var url = "/3di/wms";
  url += "?LAYERS=" + getLayer() + ":bathymetry";
  url += "&antialias=" + getAntialias();
  url += "&messages=" + getMessages();
  url += "&limits=" + info.limits[0] + "," + info.limits[1];
  bathymetry.setUrl(url);
  bathymetry.redraw();
}

function updateVelocity(){
  velocity.redraw();
  var url = "/3di/wms";
  url += "?LAYERS=" + getLayer() + ":velocity";
  url += "&time=" + getTime();
  url += "&messages=" + getMessages();
  url += "&antialias=" + getAntialias();
  url += "&nocache=yes";
  velocity.setUrl(url);
  velocity.redraw();
}

function toggleGrid(){
  var state = $("input#grid").is(":checked");
  grid.setVisibility(state);
}

function toggleQuadGrid(){
    var state = $("input#quad_grid").is(":checked");
    quad_grid.setVisibility(state);
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
function toggleVelocity(){
  var state = $("input#velocity").is(":checked");
  velocity.setVisibility(state);
}
function toggleOsm(){
  var state = $("input#osm").is(":checked");
  osm.setVisibility(state);
}
function toggleAntialias(){
  updateGrid();
  updateQuadGrid();
  updateDepth();
  updateFlood();
  updateBathymetry();
  updateVelocity();
}
function toggleWaves(){
  updateDepth();
  updateFlood();
}
  
// Slider
function slide(ui, slider){
  $("#time").text(slider.value);
  updateDepth();
  updateVelocity();
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
$("input#quad_grid").on("change", toggleQuadGrid);
$("input#depth").on("change", toggleDepth);
$("input#flood").on("change", toggleFlood);
$("input#bathymetry").on("change", toggleBathymetry);
$("input#velocity").on("change", toggleVelocity);
$("input#osm").on("change", toggleOsm);
$("input#antialias").on("change", toggleAntialias);
$("input#waves").on("change", toggleWaves);
$("#update").on("click", function(){
    updateDepth();
    console.log("updated depth");
});


toggleGrid();
toggleQuadGrid();
toggleFlood();
toggleBathymetry();
toggleVelocity();
updateLayer();

// Click handler
OpenLayers.Control.Click = OpenLayers.Class(OpenLayers.Control, {
  defaultHandlerOptions: {
    'single': true,
    'double': true,
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
        'click': this.onClick,
        'dblclick': this.onDblClick
      }, this.handlerOptions
    );
  }, 
  onClick: function(e) {
    var lonlat = map.getLonLatFromPixel(e.xy);
    lastClickPoint = [lonlat.lon, lonlat.lat];
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
  },
  onDblClick: function(e) {
    var lonlat = map.getLonLatFromPixel(e.xy);
    var wktline = 'LINESTRING (' + lastClickPoint[0].toString();
    wktline += ' ' + lastClickPoint[1].toString();
    wktline += ',' + lonlat.lon.toString();
    wktline += ' ' + lonlat.lat.toString() + ')';
    $.ajax(
      '/3di/data',
      { 
        data: {
          request: 'getprofile',
          layers: getLayer(),
          srs: map.getProjection(),
          line: wktline,
          time: getTime()
        },
        success: function(data) {console.log(data);}
      }
    );
  }
});
var click = new OpenLayers.Control.Click();
map.addControl(click);
click.activate();

