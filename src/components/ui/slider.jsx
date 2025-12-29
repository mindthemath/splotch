export function Slider({
  value,
  min,
  max,
  step,
  onValueChange,
  className = "",
  ...props
}) {
  const handleChange = (e) => {
    const newValue = parseFloat(e.target.value);
    onValueChange([newValue]);
  };

  const percentage = ((value[0] - min) / (max - min)) * 100;

  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value[0]}
      onChange={handleChange}
      className={`w-full h-2 bg-gray-200 rounded-full appearance-none cursor-pointer slider-thumb ${className}`}
      style={{
        background: `linear-gradient(to right, #000 0%, #000 ${percentage}%, #e5e7eb ${percentage}%, #e5e7eb 100%)`,
      }}
      {...props}
    />
  );
}
