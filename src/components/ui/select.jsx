import React, {
  useState,
  useRef,
  useEffect,
  createContext,
  useContext,
} from "react";

const SelectContext = createContext({
  value: null,
  onValueChange: () => {},
  isOpen: false,
  setIsOpen: () => {},
});

export function Select({ value, onValueChange, children }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <SelectContext.Provider value={{ value, onValueChange, isOpen, setIsOpen }}>
      <div className="relative">{children}</div>
    </SelectContext.Provider>
  );
}

export function SelectTrigger({ className = "", children }) {
  const { setIsOpen, isOpen } = useContext(SelectContext);

  return (
    <button
      type="button"
      className={`flex h-10 w-full items-center justify-between rounded-xl border border-gray-300 bg-white px-3 py-2 text-sm ring-offset-background placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
      onClick={() => setIsOpen(!isOpen)}
    >
      {children}
    </button>
  );
}

export function SelectValue({ placeholder }) {
  const { value } = useContext(SelectContext);

  return (
    <span>
      {value || placeholder}
      <svg
        className="ml-2 h-4 w-4 opacity-50 inline-block"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M19 9l-7 7-7-7"
        />
      </svg>
    </span>
  );
}

export function SelectContent({ children }) {
  const { isOpen, setIsOpen, onValueChange } = useContext(SelectContext);
  const contentRef = useRef(null);
  const triggerRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        contentRef.current &&
        !contentRef.current.contains(event.target) &&
        triggerRef.current &&
        !triggerRef.current.contains(event.target)
      ) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () =>
        document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [isOpen, setIsOpen]);

  if (!isOpen) return null;

  return (
    <div
      ref={contentRef}
      className="absolute z-50 min-w-[8rem] overflow-hidden rounded-xl border border-gray-300 bg-white shadow-md mt-1 w-full"
    >
      {React.Children.map(children, (child) =>
        React.cloneElement(child, {
          onClick: () => {
            const value = child.props.value;
            onValueChange(value);
            setIsOpen(false);
          },
        })
      )}
    </div>
  );
}

export function SelectItem({ value: _value, children, onClick, className = "" }) {
  return (
    <div
      className={`relative flex cursor-pointer select-none items-center rounded-lg px-2 py-1.5 text-sm outline-none hover:bg-gray-100 hover:text-gray-900 focus:bg-gray-100 focus:text-gray-900 ${className}`}
      onClick={onClick}
    >
      {children}
    </div>
  );
}
