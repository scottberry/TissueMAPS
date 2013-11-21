// FIXME decide default snapToPixel behaviour
// FIXME add option to apply snapToPixel to all coordinates?

goog.provide('ol.render.canvas.ReplayGroup');

goog.require('goog.array');
goog.require('goog.asserts');
goog.require('goog.object');
goog.require('goog.vec.Mat4');
goog.require('ol.color');
goog.require('ol.extent');
goog.require('ol.geom.flat');
goog.require('ol.render.IRender');
goog.require('ol.render.IReplayGroup');
goog.require('ol.vec.Mat4');


/**
 * @enum {number}
 */
ol.render.canvas.Instruction = {
  BEGIN_PATH: 0,
  CLOSE_PATH: 1,
  DRAW_IMAGE: 2,
  FILL: 3,
  MOVE_TO_LINE_TO: 4,
  SET_FILL_STYLE: 5,
  SET_STROKE_STYLE: 6,
  STROKE: 7
};



/**
 * @constructor
 * @implements {ol.render.IRender}
 * @protected
 */
ol.render.canvas.Replay = function() {

  /**
   * @protected
   * @type {Array.<*>}
   */
  this.instructions = [];

  /**
   * @protected
   * @type {Array.<number>}
   */
  this.coordinates = [];

  /**
   * @private
   * @type {goog.vec.Mat4.Number}
   */
  this.renderedTransform_ = goog.vec.Mat4.createNumber();

  /**
   * @private
   * @type {Array.<number>}
   */
  this.pixelCoordinates_ = [];

  /**
   * @private
   * @type {ol.Extent}
   */
  this.extent_ = ol.extent.createEmpty();

};


/**
 * @param {Array.<number>} flatCoordinates Flat coordinates.
 * @param {number} offset Offset.
 * @param {number} end End.
 * @param {number} stride Stride.
 * @param {boolean} close Close.
 * @protected
 * @return {number} My end.
 */
ol.render.canvas.Replay.prototype.appendFlatCoordinates =
    function(flatCoordinates, offset, end, stride, close) {
  var myEnd = this.coordinates.length;
  var i;
  for (i = offset; i < end; i += stride) {
    this.coordinates[myEnd++] = flatCoordinates[i];
    this.coordinates[myEnd++] = flatCoordinates[i + 1];
  }
  if (close) {
    this.coordinates[myEnd++] = flatCoordinates[offset];
    this.coordinates[myEnd++] = flatCoordinates[offset + 1];
  }
  return myEnd;
};


/**
 * @param {CanvasRenderingContext2D} context Context.
 * @param {goog.vec.Mat4.AnyType} transform Transform.
 */
ol.render.canvas.Replay.prototype.draw = function(context, transform) {
  /** @type {Array.<number>} */
  var pixelCoordinates;
  if (ol.vec.Mat4.equal2D(transform, this.renderedTransform_)) {
    pixelCoordinates = this.pixelCoordinates_;
  } else {
    pixelCoordinates = ol.geom.flat.transform2D(
        this.coordinates, 2, transform, this.pixelCoordinates_);
    goog.vec.Mat4.setFromArray(this.renderedTransform_, transform);
    goog.asserts.assert(pixelCoordinates === this.pixelCoordinates_);
  }
  var instructions = this.instructions;
  var i = 0; // instruction index
  var ii = instructions.length; // end of instructions
  var d = 0; // data index
  var dd; // end of per-instruction data
  while (i < ii) {
    var instruction = instructions[i];
    var type = /** @type {ol.render.canvas.Instruction} */ (instruction[0]);
    if (type == ol.render.canvas.Instruction.BEGIN_PATH) {
      context.beginPath();
      ++i;
    } else if (type == ol.render.canvas.Instruction.CLOSE_PATH) {
      context.closePath();
      ++i;
    } else if (type == ol.render.canvas.Instruction.DRAW_IMAGE) {
      dd = /** @type {number} */ (instruction[1]);
      var anchor = /** @type {Array.<number>} */ (instruction[2]);
      var image =  /** @type {HTMLCanvasElement|HTMLVideoElement|Image} */
          (instruction[3]);
      var snapToPixel = /** @type {boolean|undefined} */ (instruction[4]);
      for (; d < dd; d += 2) {
        var x = pixelCoordinates[d] - anchor[0];
        var y = pixelCoordinates[d + 1] - anchor[1];
        if (snapToPixel) {
          x = (x + 0.5) | 0;
          y = (y + 0.5) | 0;
        }
        context.drawImage(image, x, y);
      }
      ++i;
    } else if (type == ol.render.canvas.Instruction.FILL) {
      context.fill();
      ++i;
    } else if (type == ol.render.canvas.Instruction.MOVE_TO_LINE_TO) {
      context.moveTo(pixelCoordinates[d], pixelCoordinates[d + 1]);
      goog.asserts.assert(goog.isNumber(instruction[1]));
      dd = /** @type {number} */ (instruction[1]);
      for (d += 2; d < dd; d += 2) {
        context.lineTo(pixelCoordinates[d], pixelCoordinates[d + 1]);
      }
      ++i;
    } else if (type == ol.render.canvas.Instruction.SET_FILL_STYLE) {
      goog.asserts.assert(goog.isString(instruction[1]));
      context.fillStyle = /** @type {string} */ (instruction[1]);
      ++i;
    } else if (type == ol.render.canvas.Instruction.SET_STROKE_STYLE) {
      goog.asserts.assert(goog.isString(instruction[1]));
      goog.asserts.assert(goog.isNumber(instruction[2]));
      context.strokeStyle = /** @type {string} */ (instruction[1]);
      context.lineWidth = /** @type {number} */ (instruction[2]);
      ++i;
    } else if (type == ol.render.canvas.Instruction.STROKE) {
      context.stroke();
      ++i;
    } else {
      goog.asserts.fail();
      ++i; // consume the instruction anyway, to avoid an infinite loop
    }
  }
  // assert that all data were consumed
  goog.asserts.assert(d == pixelCoordinates.length);
  // assert that all instructions were consumed
  goog.asserts.assert(i == instructions.length);
};


/**
 * @inheritDoc
 */
ol.render.canvas.Replay.prototype.drawFeature = goog.abstractMethod;


/**
 * @inheritDoc
 */
ol.render.canvas.Replay.prototype.drawLineStringGeometry = goog.abstractMethod;


/**
 * @inheritDoc
 */
ol.render.canvas.Replay.prototype.drawMultiLineStringGeometry =
    goog.abstractMethod;


/**
 * @inheritDoc
 */
ol.render.canvas.Replay.prototype.drawPointGeometry = goog.abstractMethod;


/**
 * @inheritDoc
 */
ol.render.canvas.Replay.prototype.drawMultiPointGeometry = goog.abstractMethod;


/**
 * @inheritDoc
 */
ol.render.canvas.Replay.prototype.drawPolygonGeometry = goog.abstractMethod;


/**
 * @inheritDoc
 */
ol.render.canvas.Replay.prototype.drawMultiPolygonGeometry =
    goog.abstractMethod;


/**
 * FIXME empty description for jsdoc
 */
ol.render.canvas.Replay.prototype.finish = goog.nullFunction;


/**
 * @return {ol.Extent} Extent.
 */
ol.render.canvas.Replay.prototype.getExtent = function() {
  return this.extent_;
};


/**
 * @inheritDoc
 */
ol.render.canvas.Replay.prototype.setFillStrokeStyle = goog.abstractMethod;


/**
 * @inheritDoc
 */
ol.render.canvas.Replay.prototype.setImageStyle = goog.abstractMethod;


/**
 * @inheritDoc
 */
ol.render.canvas.Replay.prototype.setTextStyle = goog.abstractMethod;



/**
 * @constructor
 * @extends {ol.render.canvas.Replay}
 * @protected
 */
ol.render.canvas.ImageReplay = function() {

  goog.base(this);

  /**
   * @private
   * @type {Array.<number>}
   */
  this.anchor_ = null;

  /**
   * @private
   * @type {HTMLCanvasElement|HTMLVideoElement|Image}
   */
  this.image_ = null;

  /**
   * @private
   * @type {boolean|undefined}
   */
  this.snapToPixel_ = undefined;

};
goog.inherits(ol.render.canvas.ImageReplay, ol.render.canvas.Replay);


/**
 * @param {Array.<number>} flatCoordinates Flat coordinates.
 * @param {number} offset Offset.
 * @param {number} end End.
 * @param {number} stride Stride.
 * @private
 * @return {number} My end.
 */
ol.render.canvas.ImageReplay.prototype.drawCoordinates_ =
    function(flatCoordinates, offset, end, stride) {
  return this.appendFlatCoordinates(
      flatCoordinates, offset, end, stride, false);
};


/**
 * @inheritDoc
 */
ol.render.canvas.ImageReplay.prototype.drawPointGeometry =
    function(pointGeometry) {
  if (goog.isNull(this.image_)) {
    return;
  }
  goog.asserts.assert(!goog.isNull(this.anchor_));
  ol.extent.extend(this.extent_, pointGeometry.getExtent());
  var flatCoordinates = pointGeometry.getFlatCoordinates();
  var stride = pointGeometry.getStride();
  var myEnd = this.drawCoordinates_(
      flatCoordinates, 0, flatCoordinates.length, stride);
  this.instructions.push(
      [ol.render.canvas.Instruction.DRAW_IMAGE, myEnd,
       this.anchor_, this.image_, this.snapToPixel_]);
};


/**
 * @inheritDoc
 */
ol.render.canvas.ImageReplay.prototype.drawMultiPointGeometry =
    function(multiPointGeometry) {
  if (goog.isNull(this.image_)) {
    return;
  }
  goog.asserts.assert(!goog.isNull(this.anchor_));
  ol.extent.extend(this.extent_, multiPointGeometry.getExtent());
  var flatCoordinates = multiPointGeometry.getFlatCoordinates();
  var stride = multiPointGeometry.getStride();
  var myEnd = this.drawCoordinates_(
      flatCoordinates, 0, flatCoordinates.length, stride);
  this.instructions.push(
      [ol.render.canvas.Instruction.DRAW_IMAGE, myEnd,
       this.anchor_, this.image_, this.snapToPixel_]);
};


/**
 * @inheritDoc
 */
ol.render.canvas.ImageReplay.prototype.finish = function() {
  // FIXME this doesn't really protect us against further calls to draw*Geometry
  this.anchor_ = null;
  this.image_ = null;
  this.snapToPixel_ = undefined;
};


/**
 * @inheritDoc
 */
ol.render.canvas.ImageReplay.prototype.setImageStyle = function(imageStyle) {
  goog.asserts.assert(!goog.isNull(imageStyle));
  goog.asserts.assert(!goog.isNull(imageStyle.anchor));
  goog.asserts.assert(!goog.isNull(imageStyle.image));
  this.anchor_ = imageStyle.anchor;
  this.image_ = imageStyle.image;
  this.snapToPixel_ = imageStyle.snapToPixel;
};



/**
 * @constructor
 * @extends {ol.render.canvas.Replay}
 * @protected
 */
ol.render.canvas.LineStringReplay = function() {

  goog.base(this);

  /**
   * @private
   * @type {{currentStrokeStyle: (string|undefined),
   *         currentLineWidth: (number|undefined),
   *         lastStroke: number,
   *         strokeStyle: (string|undefined),
   *         lineWidth: (number|undefined)}|null}
   */
  this.state_ = {
    currentStrokeStyle: undefined,
    currentLineWidth: undefined,
    lastStroke: 0,
    strokeStyle: undefined,
    lineWidth: undefined
  };

};
goog.inherits(ol.render.canvas.LineStringReplay, ol.render.canvas.Replay);


/**
 * @param {Array.<number>} flatCoordinates Flat coordinates.
 * @param {number} offset Offset.
 * @param {number} end End.
 * @param {number} stride Stride.
 * @private
 * @return {number} end.
 */
ol.render.canvas.LineStringReplay.prototype.drawFlatCoordinates_ =
    function(flatCoordinates, offset, end, stride) {
  var state = this.state_;
  var strokeStyle = state.strokeStyle;
  var lineWidth = state.lineWidth;
  goog.asserts.assert(goog.isDef(strokeStyle));
  goog.asserts.assert(goog.isDef(lineWidth));
  if (state.currentStrokeStyle != strokeStyle ||
      state.currentLineWidth != lineWidth) {
    if (state.lastStroke != this.coordinates.length) {
      this.instructions.push([ol.render.canvas.Instruction.STROKE]);
      state.lastStroke = this.coordinates.length;
    }
    this.instructions.push(
        [ol.render.canvas.Instruction.SET_STROKE_STYLE,
         strokeStyle, lineWidth],
        [ol.render.canvas.Instruction.BEGIN_PATH]);
    state.currentStrokeStyle = strokeStyle;
    state.currentLineWidth = lineWidth;
  }
  var myEnd = this.appendFlatCoordinates(
      flatCoordinates, offset, end, stride, false);
  this.instructions.push([ol.render.canvas.Instruction.MOVE_TO_LINE_TO, myEnd]);
  return end;
};


/**
 * @inheritDoc
 */
ol.render.canvas.LineStringReplay.prototype.drawLineStringGeometry =
    function(lineStringGeometry) {
  var state = this.state_;
  goog.asserts.assert(!goog.isNull(state));
  var strokeStyle = state.strokeStyle;
  var lineWidth = state.lineWidth;
  if (!goog.isDef(strokeStyle) || !goog.isDef(lineWidth)) {
    return;
  }
  ol.extent.extend(this.extent_, lineStringGeometry.getExtent());
  var flatCoordinates = lineStringGeometry.getFlatCoordinates();
  var stride = lineStringGeometry.getStride();
  this.drawFlatCoordinates_(
      flatCoordinates, 0, flatCoordinates.length, stride);
};


/**
 * @inheritDoc
 */
ol.render.canvas.LineStringReplay.prototype.drawMultiLineStringGeometry =
    function(multiLineStringGeometry) {
  var state = this.state_;
  goog.asserts.assert(!goog.isNull(state));
  var strokeStyle = state.strokeStyle;
  var lineWidth = state.lineWidth;
  if (!goog.isDef(strokeStyle) || !goog.isDef(lineWidth)) {
    return;
  }
  ol.extent.extend(this.extent_, multiLineStringGeometry.getExtent());
  var ends = multiLineStringGeometry.getEnds();
  var flatCoordinates = multiLineStringGeometry.getFlatCoordinates();
  var stride = multiLineStringGeometry.getStride();
  var offset = 0;
  var i, ii;
  for (i = 0, ii = ends.length; i < ii; ++i) {
    offset = this.drawFlatCoordinates_(
        flatCoordinates, offset, ends[i], stride);
  }
};


/**
 * @inheritDoc
 */
ol.render.canvas.LineStringReplay.prototype.finish = function() {
  var state = this.state_;
  goog.asserts.assert(!goog.isNull(state));
  if (state.lastStroke != this.coordinates.length) {
    this.instructions.push([ol.render.canvas.Instruction.STROKE]);
  }
  this.state_ = null;
};


/**
 * @inheritDoc
 */
ol.render.canvas.LineStringReplay.prototype.setFillStrokeStyle =
    function(fillStyle, strokeStyle) {
  goog.asserts.assert(!goog.isNull(this.state_));
  goog.asserts.assert(goog.isNull(fillStyle));
  goog.asserts.assert(!goog.isNull(strokeStyle));
  goog.asserts.assert(goog.isDefAndNotNull(strokeStyle.color));
  goog.asserts.assert(goog.isDef(strokeStyle.width));
  this.state_.strokeStyle = ol.color.asString(strokeStyle.color);
  this.state_.lineWidth = strokeStyle.width;
};



/**
 * @constructor
 * @extends {ol.render.canvas.Replay}
 * @protected
 */
ol.render.canvas.PolygonReplay = function() {

  goog.base(this);

  /**
   * @private
   * @type {{currentFillColor: (string|undefined),
   *         currentStrokeStyle: (string|undefined),
   *         currentLineWidth: (number|undefined),
   *         fillStyle: (string|undefined),
   *         strokeStyle: (string|undefined),
   *         lineWidth: (number|undefined)}|null}
   */
  this.state_ = {
    currentFillColor: undefined,
    currentStrokeStyle: undefined,
    currentLineWidth: undefined,
    fillStyle: undefined,
    strokeStyle: undefined,
    lineWidth: undefined
  };

};
goog.inherits(ol.render.canvas.PolygonReplay, ol.render.canvas.Replay);


/**
 * @param {Array.<number>} flatCoordinates Flat coordinates.
 * @param {number} offset Offset.
 * @param {Array.<number>} ends Ends.
 * @param {number} stride Stride.
 * @private
 * @return {number} End.
 */
ol.render.canvas.PolygonReplay.prototype.drawFlatCoordinatess_ =
    function(flatCoordinates, offset, ends, stride) {
  var state = this.state_;
  this.instructions.push([ol.render.canvas.Instruction.BEGIN_PATH]);
  var i, ii;
  for (i = 0, ii = ends.length; i < ii; ++i) {
    var end = ends[i];
    var myEnd =
        this.appendFlatCoordinates(flatCoordinates, offset, end, stride, true);
    this.instructions.push(
        [ol.render.canvas.Instruction.MOVE_TO_LINE_TO, myEnd],
        [ol.render.canvas.Instruction.CLOSE_PATH]);
    offset = end;
  }
  // FIXME is it quicker to fill and stroke each polygon individually,
  // FIXME or all polygons together?
  if (goog.isDef(state.fillStyle)) {
    this.instructions.push([ol.render.canvas.Instruction.FILL]);
  }
  if (goog.isDef(state.strokeStyle)) {
    goog.asserts.assert(goog.isDef(state.lineWidth));
    this.instructions.push([ol.render.canvas.Instruction.STROKE]);
  }
  return offset;
};


/**
 * @inheritDoc
 */
ol.render.canvas.PolygonReplay.prototype.drawPolygonGeometry =
    function(polygonGeometry) {
  var state = this.state_;
  goog.asserts.assert(!goog.isNull(state));
  var fillStyle = state.fillStyle;
  var strokeStyle = state.strokeStyle;
  if (!goog.isDef(fillStyle) && !goog.isDef(strokeStyle)) {
    return;
  }
  if (goog.isDef(strokeStyle)) {
    goog.asserts.assert(goog.isDef(state.lineWidth));
  }
  ol.extent.extend(this.extent_, polygonGeometry.getExtent());
  this.setFillStrokeStyles_();
  var ends = polygonGeometry.getEnds();
  var flatCoordinates = polygonGeometry.getFlatCoordinates();
  var stride = polygonGeometry.getStride();
  this.drawFlatCoordinatess_(flatCoordinates, 0, ends, stride);
};


/**
 * @inheritDoc
 */
ol.render.canvas.PolygonReplay.prototype.drawMultiPolygonGeometry =
    function(multiPolygonGeometry) {
  var state = this.state_;
  goog.asserts.assert(!goog.isNull(state));
  var fillStyle = state.fillStyle;
  var strokeStyle = state.strokeStyle;
  if (!goog.isDef(fillStyle) && !goog.isDef(strokeStyle)) {
    return;
  }
  if (goog.isDef(strokeStyle)) {
    goog.asserts.assert(goog.isDef(state.lineWidth));
  }
  ol.extent.extend(this.extent_, multiPolygonGeometry.getExtent());
  this.setFillStrokeStyles_();
  var endss = multiPolygonGeometry.getEndss();
  var flatCoordinates = multiPolygonGeometry.getFlatCoordinates();
  var stride = multiPolygonGeometry.getStride();
  var offset = 0;
  var i, ii;
  for (i = 0, ii = endss.length; i < ii; ++i) {
    offset = this.drawFlatCoordinatess_(
        flatCoordinates, offset, endss[i], stride);
  }
};


/**
 * @inheritDoc
 */
ol.render.canvas.PolygonReplay.prototype.finish = function() {
  goog.asserts.assert(!goog.isNull(this.state_));
  this.state_ = null;
};


/**
 * @inheritDoc
 */
ol.render.canvas.PolygonReplay.prototype.setFillStrokeStyle =
    function(fillStyle, strokeStyle) {
  goog.asserts.assert(!goog.isNull(this.state_));
  goog.asserts.assert(!goog.isNull(fillStyle) || !goog.isNull(strokeStyle));
  if (!goog.isNull(fillStyle)) {
    goog.asserts.assert(goog.isDefAndNotNull(fillStyle.color));
    this.state_.fillStyle = ol.color.asString(fillStyle.color);
  }
  if (!goog.isNull(strokeStyle)) {
    goog.asserts.assert(goog.isDefAndNotNull(strokeStyle.color));
    goog.asserts.assert(goog.isDef(strokeStyle.width));
    this.state_.strokeStyle = ol.color.asString(strokeStyle.color);
    this.state_.lineWidth = strokeStyle.width;
  }
};


/**
 * @private
 */
ol.render.canvas.PolygonReplay.prototype.setFillStrokeStyles_ = function() {
  var state = this.state_;
  var fillStyle = state.fillStyle;
  var strokeStyle = state.strokeStyle;
  var lineWidth = state.lineWidth;
  if (goog.isDef(fillStyle) && state.currentFillColor != fillStyle) {
    this.instructions.push(
        [ol.render.canvas.Instruction.SET_FILL_STYLE, fillStyle]);
    state.currentFillColor = state.fillStyle;
  }
  if (goog.isDef(strokeStyle)) {
    goog.asserts.assert(goog.isDef(lineWidth));
    if (state.currentStrokeStyle != strokeStyle ||
        state.currentLineWidth != lineWidth) {
      this.instructions.push(
          [ol.render.canvas.Instruction.SET_STROKE_STYLE,
           strokeStyle, lineWidth]);
      state.currentStrokeStyle = strokeStyle;
      state.currentLineWidth = lineWidth;
    }
  }
};



/**
 * @constructor
 * @implements {ol.render.IReplayGroup}
 */
ol.render.canvas.ReplayGroup = function() {

  /**
   * @private
   * @type {Object.<string,
   *        Object.<ol.render.ReplayType, ol.render.canvas.Replay>>}
   */
  this.replayesByZIndex_ = {};

};


/**
 * @param {CanvasRenderingContext2D} context Context.
 * @param {ol.Extent} extent Extent.
 * @param {goog.vec.Mat4.AnyType} transform Transform.
 */
ol.render.canvas.ReplayGroup.prototype.draw =
    function(context, extent, transform) {
  /** @type {Array.<number>} */
  var zs = goog.array.map(goog.object.getKeys(this.replayesByZIndex_), Number);
  goog.array.sort(zs);
  var i, ii;
  for (i = 0, ii = zs.length; i < ii; ++i) {
    var replayes = this.replayesByZIndex_[zs[i].toString()];
    var replayType;
    for (replayType in replayes) {
      var replay = replayes[replayType];
      if (ol.extent.intersects(extent, replay.getExtent())) {
        replay.draw(context, transform);
      }
    }
  }
};


/**
 * @inheritDoc
 */
ol.render.canvas.ReplayGroup.prototype.finish = function() {
  var zKey;
  for (zKey in this.replayesByZIndex_) {
    var replayes = this.replayesByZIndex_[zKey];
    var replayKey;
    for (replayKey in replayes) {
      replayes[replayKey].finish();
    }
  }
};


/**
 * @inheritDoc
 */
ol.render.canvas.ReplayGroup.prototype.getReplay =
    function(zIndex, replayType) {
  var zIndexKey = goog.isDef(zIndex) ? zIndex.toString() : '0';
  var replayes = this.replayesByZIndex_[zIndexKey];
  if (!goog.isDef(replayes)) {
    replayes = {};
    this.replayesByZIndex_[zIndexKey] = replayes;
  }
  var replay = replayes[replayType];
  if (!goog.isDef(replay)) {
    var constructor = ol.render.canvas.BATCH_CONSTRUCTORS_[replayType];
    goog.asserts.assert(goog.isDef(constructor));
    replay = new constructor();
    replayes[replayType] = replay;
  }
  return replay;
};


/**
 * @inheritDoc
 */
ol.render.canvas.ReplayGroup.prototype.isEmpty = function() {
  return goog.object.isEmpty(this.replayesByZIndex_);
};


/**
 * @const
 * @private
 * @type {Object.<ol.render.ReplayType, function(new: ol.render.canvas.Replay)>}
 */
ol.render.canvas.BATCH_CONSTRUCTORS_ = {
  'Image': ol.render.canvas.ImageReplay,
  'LineString': ol.render.canvas.LineStringReplay,
  'Polygon': ol.render.canvas.PolygonReplay
};
